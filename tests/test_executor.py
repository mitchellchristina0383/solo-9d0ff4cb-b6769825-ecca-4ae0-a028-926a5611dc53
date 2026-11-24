"""Executor behaviour: lifetimes, views, fragmentation, failures, outputs."""

from __future__ import annotations

from memdry.memory import execute_plan
from memdry.model import parse_plan


def run(payload: dict) -> dict:
    return execute_plan(parse_plan(payload))


def test_add_and_relu_use_independent_contiguous_spans():
    result = run(
        {
            "capacity": "16",
            "inputs": "a: 1, -2, 3\nb: 10, 20, 30",
            "ops": "s = add(a, b)\nr = relu(s)",
            "outputs": "r",
        }
    )
    assert result["ok"]
    assert result["outputs"][0]["values"] == ["11", "18", "33"]
    add_step, relu_step = result["steps"][2], result["steps"][3]
    assert add_step["allocated"][0]["length"] == 3
    assert relu_step["allocated"][0]["length"] == 3
    # s's span is released as soon as relu(r) has copied out of it.
    assert [f["name"] for f in relu_step["freed"]] == ["s"]
    assert result["peak"]["elements"] == 9


def test_slice_shares_storage_and_survives_source_release():
    result = run(
        {
            "capacity": "8",
            "inputs": "x: 1, -2, 3, -4",
            "ops": "v = slice(x, 1, 4)",
            "outputs": "v",
        }
    )
    assert result["ok"]
    view = result["outputs"][0]
    assert view["role"] == "view"
    assert view["values"] == ["-2", "3", "-4"]
    assert view["storage"] == 0 and view["offset"] == 1
    slice_step = result["steps"][1]
    assert slice_step["allocated"] == []
    assert slice_step["freed"][0]["name"] == "x"
    assert slice_step["freed"][0]["storage_freed"] is False
    assert result["stats"]["allocations"] == 1


def test_view_stays_valid_when_allocator_compacts():
    result = run(
        {
            "capacity": "24",
            "inputs": (
                "X: 1, -2, 3, -4\n"
                "A: 10, -20, 30, -40, 50, -60\n"
                "Y: -1, -2, 6, 7, -8, 9"
            ),
            "ops": "v = slice(A, 1, 4)\np = relu(X)\nq = relu(Y)",
            "outputs": "v, p, q",
        }
    )
    assert result["ok"]
    assert result["stats"]["compactions"] == 1
    assert result["stats"]["moved_elements"] > 0
    outputs = {out["name"]: out for out in result["outputs"]}
    assert outputs["v"]["values"] == ["-20", "30", "-40"]
    assert outputs["v"]["role"] == "view"
    # A was allocated at offset 4; compaction moved its storage (and the view
    # inside it) to offset 0, so the view now sits at absolute offset 1.
    assert result["steps"][1]["allocated"][0]["name"] == "A"
    assert result["steps"][1]["allocated"][0]["offset"] == 4
    assert outputs["v"]["offset"] == 1
    compaction_step = next(s for s in result["steps"] if s["compaction"])
    assert compaction_step["compaction"]["moved"] >= 1
    assert outputs["q"]["values"] == ["0", "0", "6", "7", "0", "9"]


def test_fragmented_free_space_is_enough_after_compaction():
    result = run(
        {
            "capacity": "24",
            "inputs": (
                "a: 1, -4, 3, 2\n"
                "sep: 5, 5\n"
                "b: -1, 6, 7, -2\n"
                "c: 2, -3, 4, 8, -5, 1"
            ),
            "ops": "u = relu(a)\nv = relu(b)\nw = relu(c)\nz = relu(sep)",
            "outputs": "u, v, w, z",
        }
    )
    assert result["ok"]
    assert result["stats"]["compactions"] == 1
    compaction = next(step["compaction"] for step in result["steps"] if step["compaction"])
    assert compaction["before_largest"] < compaction["after_largest"]
    outputs = {out["name"]: out["values"] for out in result["outputs"]}
    assert outputs["u"] == ["1", "0", "3", "2"]
    assert outputs["v"] == ["0", "6", "7", "0"]
    assert outputs["w"] == ["2", "0", "4", "8", "0", "1"]
    assert outputs["z"] == ["5", "5"]


def test_insufficient_capacity_reports_position_and_space():
    result = run(
        {
            "capacity": "10",
            "inputs": "a: 1, 2, 3, 4\nb: 10, 20, 30, 40",
            "ops": "c = add(a, b)",
            "outputs": "c",
        }
    )
    assert not result["ok"]
    failure = result["failure"]
    assert failure["step"] == 3
    assert failure["action"]["name"] == "c"
    assert failure["needed"] == 4
    assert failure["free_total"] == 2
    assert failure["largest_free"] == 2
    assert failure["live"] == ["a", "b"]
    assert "需要 4 个连续元素" in failure["message"]
    assert result["outputs"][0]["missing"] is True


def test_add_can_trigger_compaction_and_keep_results_correct():
    result = run(
        {
            "capacity": "15",
            "inputs": (
                "p0: 0, 1, 5\np1: 5, 2\np2: 0, 2\np3: 2, 0, -3"
            ),
            "ops": (
                "q0 = add(p2, p1)\nq1 = add(p0, p3)\nq2 = add(p1, q0)\n"
                "q3 = add(p3, p0)\nq4 = add(p2, q2)"
            ),
            "outputs": "q3",
        }
    )
    assert result["ok"]
    assert result["stats"]["compactions"] == 1
    compaction_step = next(step for step in result["steps"] if step["compaction"])
    assert compaction_step["title"].startswith("执行 q3 = add")
    assert compaction_step["compaction"]["before_largest"] == 2
    assert compaction_step["compaction"]["after_largest"] == 5
    assert result["outputs"][0]["values"] == ["2", "1", "2"]


def test_inputs_are_released_at_last_use_but_outputs_are_kept():
    result = run(
        {
            "capacity": "20",
            "inputs": "a: 1, 2\nb: 3, 4",
            "ops": "s = add(a, b)\nt = add(s, a)",
            "outputs": "t",
        }
    )
    assert result["ok"]
    add_step, second_step = result["steps"][2], result["steps"][3]
    # s is still needed by the second add, so the first releases b only.
    assert [f["name"] for f in add_step["freed"]] == ["b"]
    assert [f["name"] for f in second_step["freed"]] == ["s", "a"]
    assert [out["name"] for out in result["outputs"]] == ["t"]
    assert result["outputs"][0]["values"] == ["5", "8"]


def test_shared_storage_freed_only_when_source_and_view_are_done():
    result = run(
        {
            "capacity": "12",
            "inputs": "x: 1, 2, 3, 4",
            "ops": "v = slice(x, 0, 2)\ns = add(v, v)",
            "outputs": "s",
        }
    )
    assert result["ok"]
    slice_step, add_step = result["steps"][1], result["steps"][2]
    # x's handle is released at the slice, but the block survives for the view.
    assert slice_step["freed"] == [
        {"name": "x", "role": "input", "storage": 0, "storage_freed": False}
    ]
    # Once the view is consumed too, the block is finally released.
    assert add_step["freed"] == [
        {"name": "v", "role": "view", "storage": 0, "storage_freed": True}
    ]
    assert add_step["free"] == 12 - 2
    assert result["outputs"][0]["values"] == ["2", "4"]


def test_compaction_can_move_a_storage_that_holds_a_live_view():
    result = run(
        {
            "capacity": "10",
            "inputs": (
                "n0: -2, -1\n"
                "n1: -5, -4, 3, -4, 4, -2\n"
                "n2: 2, 0, 7, 7\n"
                "n3: 9, 8, -1, 5, 7"
            ),
            "ops": "v = slice(n3, 1, 4)\nr0 = relu(v)\nr1 = relu(n0)\nr2 = relu(n3)",
            "outputs": "n3, v",
        }
    )
    assert result["ok"]
    assert result["stats"]["compactions"] == 1
    # n3 was allocated at offset 2 and moved to 0 together with its view.
    assert result["steps"][3]["allocated"][0]["offset"] == 2
    outputs = {out["name"]: out for out in result["outputs"]}
    assert outputs["n3"]["values"] == ["9", "8", "-1", "5", "7"]
    assert outputs["v"]["values"] == ["8", "-1", "5"]
    assert outputs["v"]["storage"] == 3
    assert outputs["v"]["offset"] == 1
    assert result["layout"][0]["start"] == 0
    assert result["layout"][0]["views"][0]["name"] == "v"


def test_unused_value_is_released_immediately():
    result = run(
        {
            "capacity": "12",
            "inputs": "a: 1, 2\nb: 3, 4\nz: 9, 9",
            "ops": "s = add(a, b)",
            "outputs": "s",
        }
    )
    assert result["ok"]
    z_step = result["steps"][2]
    assert [f["name"] for f in z_step["freed"]] == ["z"]
    assert any("z" in warning for warning in result["warnings"])


def test_slice_of_slice_and_negative_indices():
    result = run(
        {
            "capacity": "12",
            "inputs": "x: 1, 2, 3, 4, 5, 6",
            "ops": "v = slice(x, -4, -1)\nw = slice(v, 1, 3)",
            "outputs": "w",
        }
    )
    assert result["ok"]
    assert result["outputs"][0]["values"] == ["4", "5"]
    assert result["outputs"][0]["offset"] == 3


def test_float32_rounding_and_overflow_are_exposed():
    result = run(
        {
            "capacity": "8",
            "inputs": "a: 0.1, 3.4e38\nb: 0.2, 3.4e38",
            "ops": "s = add(a, b)",
            "outputs": "s",
        }
    )
    assert result["ok"]
    assert result["outputs"][0]["values"][0] == "0.30000001"
    assert result["outputs"][0]["values"][1] == "inf"
    assert any("inf" in warning for warning in result["warnings"])
