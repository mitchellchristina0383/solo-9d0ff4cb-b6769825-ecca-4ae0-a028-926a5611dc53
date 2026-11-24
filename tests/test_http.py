"""Plan parsing and HTTP surface tests."""

from __future__ import annotations

import pytest

from memdry.app import create_app
from memdry.model import ValidationError, parse_plan


@pytest.fixture()
def client():
    app = create_app({"TESTING": True})
    return app.test_client()


def test_parse_rejects_bad_capacity():
    with pytest.raises(ValidationError) as info:
        parse_plan({"capacity": "0", "inputs": "a: 1", "ops": "b = relu(a)", "outputs": "b"})
    assert any("大于 0" in message for message in info.value.messages)


def test_parse_rejects_add_length_mismatch():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2\nb: 1",
                "ops": "c = add(a, b)",
                "outputs": "c",
            }
        )
    assert any("等长" in message for message in info.value.messages)


def test_parse_rejects_stepped_slice_and_unknown_operands():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = slice(a, 0, 3, 2)",
                "outputs": "v",
            }
        )
    assert any("步长" in message for message in info.value.messages)

    with pytest.raises(ValidationError):
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = relu(nope)",
                "outputs": "v",
            }
        )


def test_parse_rejects_empty_slice_and_duplicate_names():
    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2, 3",
                "ops": "v = slice(a, 2, 2)",
                "outputs": "v",
            }
        )
    assert any("为空" in message for message in info.value.messages)

    with pytest.raises(ValidationError) as info:
        parse_plan(
            {
                "capacity": "10",
                "inputs": "a: 1, 2\nb: 1, 2",
                "ops": "b = relu(a)",
                "outputs": "b",
            }
        )
    assert any("重名" in message for message in info.value.messages)


def test_index_page_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    body = response.get_data(as_text=True)
    assert "端侧推理" in body
    assert "缓冲容量" in body


def test_healthz(client):
    assert client.get("/healthz").get_json() == {"ok": True}


def test_api_requires_json(client):
    response = client.post("/api/run", data="not json", content_type="text/plain")
    assert response.status_code == 400
    assert response.get_json()["ok"] is False


def test_api_returns_validation_errors(client):
    response = client.post(
        "/api/run",
        json={"capacity": "2", "inputs": "a: 1, 2", "ops": "b = relu(nope)", "outputs": "b"},
    )
    assert response.status_code == 400
    assert response.get_json()["errors"]


def test_api_success_payload(client):
    response = client.post(
        "/api/run",
        json={
            "capacity": "12",
            "inputs": "a: 1, 2, 3\nb: 4, 5, 6",
            "ops": "s = add(a, b)\nv = slice(s, 1, 3)",
            "outputs": "v",
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is True
    assert body["outputs"][0]["values"] == ["7", "9"]
    assert body["peak"]["elements"] == 9
    assert body["layout"]


def test_api_insufficient_capacity_is_a_200_report(client):
    response = client.post(
        "/api/run",
        json={
            "capacity": "4",
            "inputs": "a: 1, 2\nb: 3, 4",
            "ops": "s = add(a, b)",
            "outputs": "s",
        },
    )
    assert response.status_code == 200
    body = response.get_json()
    assert body["ok"] is False
    assert body["failure"]["needed"] == 2
    assert body["failure"]["step"] == 3
