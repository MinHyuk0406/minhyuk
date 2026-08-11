"""Small shared HTTP helpers for Vercel's Python Functions.

The static dashboard remains at the project root. Vercel maps api/scenario.py
and api/coach.py to their matching /api routes.
"""

from __future__ import annotations

import json
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler
from typing import Any


MAX_REQUEST_BYTES = 25_000


class ApiHandler(BaseHTTPRequestHandler):
    """JSON-only base handler shared by the two serverless routes."""

    def json_response(self, payload: dict[str, Any], status: HTTPStatus = HTTPStatus.OK) -> None:
        encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def read_json(self) -> dict[str, Any]:
        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > MAX_REQUEST_BYTES:
            raise ValueError("요청 크기가 올바르지 않습니다.")
        return json.loads(self.rfile.read(content_length).decode("utf-8"))

    def method_not_allowed(self) -> None:
        self.json_response({"error": "POST 요청만 지원합니다."}, HTTPStatus.METHOD_NOT_ALLOWED)
