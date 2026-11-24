"""Memory dry run: parse a plan, simulate one fixed float32 buffer, report."""

from __future__ import annotations

import os

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .memory import execute_plan
from .model import ValidationError, parse_plan

MAX_BODY_BYTES = 2 * 1024 * 1024


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
        except ValidationError as exc:
            return jsonify({"ok": False, "errors": exc.messages}), 400
        result = execute_plan(plan)
        return jsonify(result), 200

    @app.errorhandler(413)
    def too_large(_exc):
        return (
            jsonify(
                {
                    "ok": False,
                    "errors": [
                        f"请求体超过 {MAX_BODY_BYTES // 1024} KiB 上限，请缩短输入张量"
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


app = create_app()


if __name__ == "__main__":
    app.run(
        host=os.environ.get("HOST", "127.0.0.1"),
        port=int(os.environ.get("PORT", "8000")),
        debug=os.environ.get("FLASK_DEBUG", "0") == "1",
    )
