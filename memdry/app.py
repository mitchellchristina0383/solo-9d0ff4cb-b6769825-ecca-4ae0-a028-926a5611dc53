"""Memory dry run: parse a plan, simulate one fixed float32 buffer, report."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .handoff import execute_handoff, load_handoff
from .memory import execute_plan
from .model import ValidationError, parse_plan

# Handoff packages carry base64 float32 payloads; allow headroom beyond a plain plan.
MAX_BODY_BYTES = 8 * 1024 * 1024


def create_app(test_config: dict | None = None) -> Flask:
    app = Flask(
        __name__,
        static_folder="../static",
        template_folder="../templates",
    )
    app.config.update(MAX_CONTENT_LENGTH=MAX_BODY_BYTES)
    if test_config:
        app.config.update(test_config)

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/healthz")
    def healthz():
        return jsonify({"ok": True})

    @app.post("/api/run")
    def run_plan():
        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "ok": False,
                        "errors": [
                            "请求体必须是 JSON（Content-Type: application/json）"
                        ],
                    }
                ),
                400,
            )
        try:
            plan = parse_plan(payload)
            stop_after = _parse_split_after(payload, len(plan.ops))
        except ValidationError as exc:
            return jsonify({"ok": False, "errors": exc.messages}), 400
        result = execute_plan(plan, stop_after=stop_after)
        return jsonify(result), 200

    @app.post("/api/handoff/run")
    def run_handoff():
        capacity, package_data = _read_handoff_request()
        try:
            if package_data is None:
                raise ValidationError("缺少交接包文件（字段名 package，类型为 .json）")
            handoff = load_handoff(package_data)
            if capacity is None:
                raise ValidationError("后段缓冲容量不能为空（按元素计）")
            if capacity < 1:
                raise ValidationError("后段缓冲容量必须大于 0")
            if capacity > 1_000_000:
                raise ValidationError("后段缓冲容量过大（上限 1000000 个元素）")
        except ValidationError as exc:
            return jsonify({"ok": False, "errors": exc.messages}), 400
        result = execute_handoff(handoff, capacity)
        return jsonify(result), 200

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            jsonify(
                {
                    "ok": False,
                    "errors": [
                        f"请求体超过 {MAX_BODY_BYTES // 1024 // 1024} MiB 上限，"
                        "请缩短输入张量或交接包"
                    ],
                }
            ),
            413,
        )

    @app.errorhandler(Exception)
    def internal_error(exc):
        if isinstance(exc, ValidationError):
            return jsonify({"ok": False, "errors": exc.messages}), 400
        if isinstance(exc, HTTPException):
            if request.path.startswith("/api/"):
                return (
                    jsonify(
                        {
                            "ok": False,
                            "errors": [
                                f"{exc.code} {exc.name}：{request.path} 不接受该请求"
                            ],
                        }
                    ),
                    exc.code,
                )
            return exc
        app.logger.exception("unexpected error")
        return (
            jsonify({"ok": False, "errors": ["服务器内部错误，请检查输入后重试"]}),
            500,
        )

    return app


def _parse_split_after(payload: dict, op_count: int) -> int | None:
    """Normalize the optional front-segment boundary (op index, zero based)."""
    raw = payload.get("split_after")
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        value = int(str(raw).strip(), 10)
    except (ValueError, TypeError):
        raise ValidationError(f"分界算子序号必须是整数，收到 {raw!r}")
    if value < 0 or value >= op_count:
        raise ValidationError(
            f"分界序号 {value} 超出算子范围（0 到 {op_count - 1}，共 {op_count} 个算子）"
        )
    if value == op_count - 1:
        raise ValidationError("分界不能选最后一个算子：后段至少需要保留一个算子")
    return value

def _read_handoff_request():
    """Accept either multipart/form-data (file upload) or JSON with an inline package."""
    content_type = request.mimetype or ""
    if content_type.startswith("multipart/form-data"):
        capacity_raw = request.form.get("capacity", "")
        capacity = _parse_optional_capacity(capacity_raw)
        upload = request.files.get("package")
        if upload is None:
            return capacity, None
        try:
            return capacity, _decode_package(upload.read())
        except ValidationError:
            raise
    payload = request.get_json(silent=True)
    if payload is None:
        raise ValidationError(
            "请求必须是 multipart 表单（package 文件 + capacity）或 JSON 对象"
        )
    capacity = _parse_optional_capacity(payload.get("capacity", ""))
    package_data = payload.get("package")
    if isinstance(package_data, str):
        package_data = _decode_package(package_data.encode("utf-8"))
    return capacity, package_data


def _parse_optional_capacity(raw) -> int | None:
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    try:
        return int(str(raw).strip(), 10)
    except (ValueError, TypeError):
        raise ValidationError(f"后段缓冲容量必须是整数，收到 {raw!r}")


def _decode_package(raw: bytes):
    import json

    if not raw:
        raise ValidationError("交接包文件为空")
    try:
        return json.loads(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise ValidationError("交接包必须是 UTF-8 编码的 JSON 文件") from exc
    except json.JSONDecodeError as exc:
        raise ValidationError(f"交接包不是合法 JSON：第 {exc.lineno} 行第 {exc.colno} 列") from exc


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
