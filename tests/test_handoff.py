"""Split-run handoff: boundary snapshots, visible-element transfer, resume."""

from __future__ import annotations

import base64
import io
import json
import struct

import pytest

from memdry.app import create_app
from memdry.handoff import HANDOFF_FORMAT, execute_handoff, load_handoff
from memdry.memory import execute_plan
from memdry.model import ValidationError, parse_plan

OVERLAP_PLAN = {
    "capacity": "16",
    "inputs": "X: 1, -2, 3, -4, 5, -6, 7, -8\nY: 10, 20, 30, 40",
    "ops": (
        "v1 = slice(X, 1, 5)\n"
        "v2 = slice(X, 3, 7)\n"
        "r = relu(v1)\n"
        "s = add(v2, Y)"
    ),
    "outputs": "r, s",
}


def split_run(payload: dict, split_after: int, back_capacity: int):
    front = execute_plan(parse_plan(payload), stop_after=split_after)
    assert front["ok"], front.get("failure")
    package = front["handoff"]["package"]
    back = execute_handoff(load_handoff(package), back_capacity)
    return front, package, back


def test_overlapping_slices_transfer_union_only_once():
    front, package, back = split_run(OVERLAP_PLAN, 1, 24)
    # v1 covers X[1:5], v2 covers X[3:7]; the union is X[1:7] = 6 elements,
    # even though the two views name 8 element references in total.
    x_storage = next(s for s in package["storages"] if s["origin_sid"] == 0)
    assert x_storage["length"] == 8
    assert [(seg["start"], seg["length"]) for seg in x_storage["segments"]] == [(1, 6)]
    assert front["handoff"]["transferred_elements"] == 10  # 6 (X block) + 4 (Y)
    # The dead source X is carried as metadata, holding no back-side reference.
    x_meta = next(t for t in x_storage["tensors"] if t["name"] == "X")
    assert x_meta["released"] is True
    v1_meta = next(t for t in x_storage["tensors"] if t["name"] == "v1")
    assert v1_meta["released"] is False and v1_meta["slice_of"] == "X"
    assert back["ok"]


def test_back_outputs_match_a_single_full_run_bit_for_bit():
    # The full run needs the same peak (20) that the back segment reaches.
    full_plan = dict(OVERLAP_PLAN, capacity="24")
    whole = execute_plan(parse_plan(full_plan))
    _, _, back = split_run(OVERLAP_PLAN, 1, 24)
    whole_values = {o["name"]: o["values"] for o in whole["outputs"]}
    back_values = {o["name"]: o["values"] for o in back["outputs"]}
    assert back_values == whole_values
    assert back_values["r"] == ["0", "3", "0", "5"]
    assert back_values["s"] == ["6", "25", "24", "47"]


def test_source_storage_allocated_once_and_shared_by_slices():
    _, package, back = split_run(OVERLAP_PLAN, 1, 24)
    load_steps = [step for step in back["steps"] if step["kind"] == "load"]
    assert len(load_steps) == len(package["storages"]) == 2
    # The X block is one allocation of the original length 8 shared by v1/v2.
    x_load = next(step for step in load_steps if "来源存储 #0" in step["title"])
    assert x_load["allocated"][0]["length"] == 8
    handles = {o["name"]: o for o in back["outputs"]}
    assert handles  # outputs render normally


def test_storage_freed_only_after_last_reference():
    # v is consumed by the add; afterwards the shared block is released once.
    payload = {
        "capacity": "12",
        "inputs": "x: 1, 2, 3, 4",
        "ops": "v = slice(x, 0, 2)\ns = add(v, v)",
        "outputs": "s",
    }
    front, package, back = split_run(payload, 0, 12)
    # The package keeps the x block (x itself released, v alive).
    assert package["storages"][0]["length"] == 4
    add_step = next(step for step in back["steps"] if step["kind"] == "op")
    assert [f["name"] for f in add_step["freed"]] == ["v"]
    assert add_step["freed"][0]["storage_freed"] is True
    assert back["outputs"][0]["values"] == ["2", "4"]


def test_float32_bit_pattern_survives_base64_roundtrip():
    # -0.0 crosses the boundary inside a retained slice; the add overflows to
    # real +inf (3.4e38 itself rounds to the largest finite binary32, not inf).
    payload = {
        "capacity": "16",
        "inputs": "a: -0.0, 1.0, 3.4e38\nb: 1.0, 2.0, 3.4e38",
        "ops": "c = add(a, b)\nv = slice(a, 0, 1)\nd = relu(c)",
        "outputs": "v, c, d",
    }
    whole = execute_plan(parse_plan(payload))
    front, package, back = split_run(payload, 0, 16)
    # The a block survives because view v keeps it; the first packed bit is -0.0.
    a_storage = next(s for s in package["storages"] if s["origin_sid"] == 0)
    a_raw = base64.b64decode(a_storage["segments"][0]["values_b64"], validate=True)
    assert struct.unpack("<I", a_raw[:4])[0] == 0x80000000
    # The add-result storage carries the overflow-generated +inf bit pattern.
    c_storage = next(
        s for s in package["storages"]
        if any(t["name"] == "c" for t in s["tensors"])
    )
    c_raw = base64.b64decode(c_storage["segments"][0]["values_b64"], validate=True)
    assert struct.unpack("<I", c_raw[8:12])[0] == 0x7F800000
    whole_values = {o["name"]: o["values"] for o in whole["outputs"]}
    back_values = {o["name"]: o["values"] for o in back["outputs"]}
    assert back_values == whole_values
    assert back_values["v"] == ["-0"]
    assert back_values["c"] == ["1", "3", "inf"]
    assert back_values["d"] == ["1", "3", "inf"]


def test_back_capacity_failure_distinguishes_load_from_op():
    front = execute_plan(parse_plan(OVERLAP_PLAN), stop_after=1)
    package = front["handoff"]["package"]
    handoff = load_handoff(package)

    load_fail = execute_handoff(handoff, 6)
    assert not load_fail["ok"]
    failure = load_fail["failure"]
    assert failure["phase"] == "load"
    assert failure["phase_label"] == "载入交接包"
    assert failure["action"]["kind"] == "load"
    assert failure["action"]["length"] == 8  # original X block length
    assert failure["needed"] == 8

    # At 16 the two source blocks (12) plus r (4) fill the buffer; the next
    # operator s cannot allocate, and the report names that operator exactly.
    op_fail = execute_handoff(load_handoff(package), 16)
    assert not op_fail["ok"]
    op_failure = op_fail["failure"]
    assert op_failure["phase"] == "op"
    assert op_failure["phase_label"] == "执行后段算子"
    assert op_failure["action"]["kind"] == "add"
    assert op_failure["action"]["name"] == "s"
    assert op_failure["needed"] == 4

    # The same package succeeds once the capacity is large enough.
    ok = execute_handoff(load_handoff(package), 24)
    assert ok["ok"]
    assert ok["peak"]["elements"] == 20


def test_retained_front_output_is_part_of_the_package():
    payload = {
        "capacity": "16",
        "inputs": "a: 1, 2, 3",
        "ops": "p = relu(a)\nq = relu(p)",
        "outputs": "p, q",
    }
    front, package, back = split_run(payload, 0, 16)
    names = {t["name"] for s in package["storages"] for t in s["tensors"]}
    assert "p" in names  # retained even though the back op only reads p once
    values = {o["name"]: o["values"] for o in back["outputs"]}
    assert values["p"] == ["1", "2", "3"]
    assert values["q"] == ["1", "2", "3"]


def test_load_handoff_rejects_tampered_documents():
    front = execute_plan(parse_plan(OVERLAP_PLAN), stop_after=1)
    package = front["handoff"]["package"]

    def reject(tampered, fragment):
        with pytest.raises(ValidationError) as info:
            load_handoff(tampered)
        assert any(fragment in message for message in info.value.messages), info.value.messages

    bad_format = dict(package, format="other/1")
    reject(bad_format, "格式标识")

    overlap = json.loads(json.dumps(package))
    overlap["storages"][0]["segments"].append(dict(overlap["storages"][0]["segments"][0]))
    reject(overlap, "重叠")

    missing = json.loads(json.dumps(package))
    target = missing["storages"][0]["tensors"][0]
    target["rel_offset"] = 99
    reject(missing, "超出存储块")

    unknown_ref = json.loads(json.dumps(package))
    unknown_ref["back_ops"][0]["operands"] = ["ghost"]
    reject(unknown_ref, "未定义")

    not_jsonish = {"format": HANDOFF_FORMAT, "split_after": 0}
    reject(not_jsonish, "split_op")


def test_split_after_last_op_is_rejected():
    plan = parse_plan(OVERLAP_PLAN)
    with pytest.raises(ValidationError) as info:
        # Mirrors the app-level boundary check.
        from memdry.app import _parse_split_after

        _parse_split_after({"split_after": len(plan.ops) - 1}, len(plan.ops))
    assert any("后段至少需要保留一个算子" in m for m in info.value.messages)

# -- HTTP surface ------------------------------------------------------------

@pytest.fixture()
def client():
    return create_app({"TESTING": True}).test_client()


def test_http_front_run_returns_downloadable_package(client):
    response = client.post("/api/run", json={**OVERLAP_PLAN, "split_after": "1"})
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] and body["handoff"]["format"] == HANDOFF_FORMAT
    assert body["handoff"]["transferred_elements"] == 10
    # The package itself is self-contained JSON.
    package = body["handoff"]["package"]
    json.dumps(package)
    assert package["split_op"]["name"] == "v2"


def test_http_handoff_upload_and_resume(client):
    front = client.post("/api/run", json={**OVERLAP_PLAN, "split_after": "1"}).get_json()
    package_bytes = json.dumps(front["handoff"]["package"]).encode("utf-8")

    response = client.post(
        "/api/handoff/run",
        data={
            "capacity": "24",
            "package": (io.BytesIO(package_bytes), "handoff.json"),
        },
        content_type="multipart/form-data",
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"]
    values = {o["name"]: o["values"] for o in body["outputs"]}
    assert values == {"r": ["0", "3", "0", "5"], "s": ["6", "25", "24", "47"]}
    assert body["handoff_meta"]["transferred_elements"] == 10
    assert body["steps"][0]["kind"] == "load"


def test_http_handoff_capacity_failure_and_bad_upload(client):
    front = client.post("/api/run", json={**OVERLAP_PLAN, "split_after": "1"}).get_json()
    package_bytes = json.dumps(front["handoff"]["package"]).encode("utf-8")

    short = client.post(
        "/api/handoff/run",
        data={"capacity": "6", "package": (io.BytesIO(package_bytes), "h.json")},
        content_type="multipart/form-data",
    )
    assert short.status_code == 200
    assert short.get_json()["failure"]["phase"] == "load"

    bad = client.post(
        "/api/handoff/run",
        data={"capacity": "24", "package": (io.BytesIO(b"nope"), "h.json")},
        content_type="multipart/form-data",
    )
    assert bad.status_code == 400
    assert bad.get_json()["errors"]

    missing_file = client.post(
        "/api/handoff/run",
        data={"capacity": "24"},
        content_type="multipart/form-data",
    )
    assert missing_file.status_code == 400
    assert any("交接包" in m for m in missing_file.get_json()["errors"])


def test_http_plain_run_has_no_handoff_payload(client):
    response = client.post("/api/run", json=OVERLAP_PLAN)
    assert response.status_code == 200
    assert "handoff" not in response.get_json()
