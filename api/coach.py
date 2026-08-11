"""Vercel Function: POST /api/coach."""

from __future__ import annotations

import json
import sys
from http import HTTPStatus
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import generate_coach, scenario_payload  # noqa: E402
from vercel_api import ApiHandler  # noqa: E402


class handler(ApiHandler):  # Vercel discovers this class by name.
    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        try:
            self.json_response(generate_coach(scenario_payload(self.read_json())))
        except json.JSONDecodeError:
            self.json_response({"error": "JSON 요청 형식이 올바르지 않습니다."}, HTTPStatus.BAD_REQUEST)
        except ValueError as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self.json_response({"error": "코치 분석을 처리하지 못했습니다. 입력값을 다시 확인하세요."}, HTTPStatus.INTERNAL_SERVER_ERROR)

    def do_GET(self) -> None:  # noqa: N802
        self.method_not_allowed()
