"""Serve the OpenSafe AI dashboard, scenario API, and optional GenAI coach."""

from __future__ import annotations

import argparse
import json
import os
import urllib.error
import urllib.request
import webbrowser
from functools import lru_cache
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import quote, urlparse


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "dashboard-data.json"
MAX_REQUEST_BYTES = 25_000


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def median(values: list[float]) -> float:
    ordered = sorted(values)
    midpoint = len(ordered) // 2
    return ordered[midpoint] if len(ordered) % 2 else (ordered[midpoint - 1] + ordered[midpoint]) / 2


def level(score: float) -> str:
    if score >= 75:
        return "운영 여력 높음"
    if score >= 60:
        return "조건부 가능"
    if score >= 40:
        return "보완 필요"
    return "재검토 필요"


def number(payload: dict[str, Any], field: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(payload.get(field, default))
    except (TypeError, ValueError):
        value = default
    return clamp(value, minimum, maximum)


@lru_cache(maxsize=1)
def load_data() -> dict[str, Any]:
    with DATA_PATH.open("r", encoding="utf-8") as source:
        return json.load(source)


def find_record(dong_code: str, industry_code: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    data = load_data()
    industry_rows = [row for row in data["records"] if row["ic"] == industry_code]
    record = next((row for row in industry_rows if row["dc"] == dong_code), None)
    if not record:
        raise ValueError("선택한 행정동·업종 조합의 분석 데이터가 없습니다.")
    return record, industry_rows


def scenario_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Calculate an auditable operating scenario from market data and user inputs.

    Currency inputs are in 10,000 KRW (만원) to keep the interface readable.
    This is a break-even scenario, not an individual-business prediction.
    """
    dong_code = str(payload.get("dong_code", ""))
    industry_code = str(payload.get("industry_code", ""))
    record, industry_rows = find_record(dong_code, industry_code)

    rent = number(payload, "rent_manwon", 250, 0, 20_000)
    other_fixed = number(payload, "other_fixed_manwon", 350, 0, 20_000)
    variable_rate = number(payload, "variable_cost_rate", 35, 1, 95)
    basket_won = number(payload, "basket_won", 9_000, 1_000, 1_000_000)
    operating_days = number(payload, "operating_days", 26, 1, 31)
    sales_adjustment = number(payload, "sales_adjustment", 0, -50, 50)
    startup_budget = number(payload, "startup_budget_manwon", 8_000, 0, 500_000)

    base_monthly_revenue = record["ss"] / 3 / 10_000
    monthly_revenue = base_monthly_revenue * (1 + sales_adjustment / 100)
    contribution_margin = 1 - variable_rate / 100
    gross_profit = monthly_revenue * contribution_margin
    fixed_cost = rent + other_fixed
    monthly_profit = gross_profit - fixed_cost
    break_even_revenue = fixed_cost / contribution_margin
    transactions_per_day = monthly_revenue * 10_000 / basket_won / operating_days
    runway_months = None if monthly_profit >= 0 else startup_budget / abs(monthly_profit)

    median_sales_per_store = median([row["ss"] for row in industry_rows])
    market_score = 100 - record["mr"]
    demand_score = clamp((record["ss"] / median_sales_per_store) * 50, 0, 100)
    profit_score = clamp(50 + monthly_profit / max(fixed_cost, 1) * 120, 0, 100)
    runway_score = 100 if monthly_profit >= 0 else clamp((runway_months or 0) / 12 * 100, 0, 100)
    resilience = round(0.50 * profit_score + 0.20 * market_score + 0.20 * demand_score + 0.10 * runway_score)

    return {
        "market": {
            "dong": record["d"],
            "industry": record["i"],
            "current_risk": record["r"],
            "current_risk_label": record["rl"],
            "ai_next_close_rate": record["mp"],
            "ai_range_low": record["mlow"],
            "ai_range_high": record["mhigh"],
            "ai_relative_risk": record["ml"],
            "sales_per_store_quarterly_won": record["ss"],
            "median_sales_per_store_quarterly_won": round(median_sales_per_store),
            "close_rate": record["cr"],
            "store_count": record["s"],
        },
        "inputs": {
            "rent_manwon": rent,
            "other_fixed_manwon": other_fixed,
            "variable_cost_rate": variable_rate,
            "basket_won": basket_won,
            "operating_days": operating_days,
            "sales_adjustment": sales_adjustment,
            "startup_budget_manwon": startup_budget,
        },
        "scenario": {
            "base_monthly_revenue_manwon": round(base_monthly_revenue),
            "monthly_revenue_manwon": round(monthly_revenue),
            "gross_profit_manwon": round(gross_profit),
            "fixed_cost_manwon": round(fixed_cost),
            "monthly_profit_manwon": round(monthly_profit),
            "break_even_revenue_manwon": round(break_even_revenue),
            "transactions_per_day": round(transactions_per_day, 1),
            "runway_months": None if runway_months is None else round(runway_months, 1),
        },
        "resilience": {
            "score": resilience,
            "label": level(resilience),
            "components": {
                "profitability": round(profit_score),
                "market_outlook": round(market_score),
                "demand_position": round(demand_score),
                "financial_cushion": round(runway_score),
            },
        },
        "disclaimer": "이는 행정동·업종 평균 매출과 사용자가 입력한 가정으로 계산한 손익 시나리오입니다. 개별 점포의 실제 매출·생존을 예측하거나 보장하지 않습니다.",
    }


def rule_based_coach(context: dict[str, Any]) -> dict[str, Any]:
    """A useful local fallback when no API key is configured."""
    market, inputs, scenario, resilience = (
        context["market"],
        context["inputs"],
        context["scenario"],
        context["resilience"],
    )
    priorities: list[dict[str, str]] = []
    checks: list[str] = []
    if scenario["monthly_profit_manwon"] < 0:
        priorities.append({
            "title": "손익분기 구조 보완",
            "why": f"현재 가정에서는 월 {abs(scenario['monthly_profit_manwon']):,}만 원 적자입니다.",
            "action": f"월매출을 최소 {scenario['break_even_revenue_manwon']:,}만 원으로 올리거나 고정비를 낮추는 조건을 계약 전에 검토하세요.",
        })
        checks.append("동일 시간대의 실제 객수와 객단가로 손익분기 매출을 다시 계산하세요.")
    else:
        priorities.append({
            "title": "흑자 가정의 현장 검증",
            "why": f"현재 가정에서는 월 {scenario['monthly_profit_manwon']:,}만 원의 운영 잉여가 계산됩니다.",
            "action": "평일·주말 각각 3회 이상 유동과 경쟁점 객수를 조사해 매출 가정이 재현되는지 확인하세요.",
        })
    if market["ai_relative_risk"] in {"높음", "주의"}:
        priorities.append({
            "title": "시장 변동성 대비",
            "why": f"AI 다음 분기 폐업률 예측은 {market['ai_next_close_rate']}%({market['ai_range_low']}~{market['ai_range_high']}%)입니다.",
            "action": "권리금·인테리어 비용을 단계적으로 집행하고, 90일 성과 기준을 정해 두세요.",
        })
        checks.append("최근 폐업 점포의 위치·업종·가격대를 직접 확인하세요.")
    if inputs["sales_adjustment"] < 0:
        checks.append("보수 매출 시나리오가 임대료·인건비를 감당하는지 확인하세요.")
    else:
        checks.append("매출 가정에 배달·포장 매출이 포함되는지 구분하세요.")
    checks.append("계약 전 월 임대료 외 관리비, 보증금 이자, 소모품 비용을 고정비에 반영하세요.")
    return {
        "summary": f"{market['dong']} {market['industry']}의 12개월 운영 여력은 {resilience['score']}점({resilience['label']})입니다.",
        "decision": "계약 판단 전 현장 관측과 임대차 조건 검증이 필요한 시나리오입니다.",
        "evidence": [
            {"label": "AI 다음 분기 폐업률", "value": f"{market['ai_next_close_rate']}%", "interpretation": "행정동·업종 단위의 다음 분기 상대 위험 신호입니다."},
            {"label": "점포당 분기 추정매출", "value": f"{market['sales_per_store_quarterly_won'] / 10_000:,.0f}만 원", "interpretation": "동일 업종 행정동 평균과 비교하는 시장 수요 참고값입니다."},
            {"label": "시나리오 월 손익", "value": f"{scenario['monthly_profit_manwon']:,}만 원", "interpretation": "사용자가 입력한 비용·매출 가정으로 계산한 결과입니다."},
        ],
        "priorities": priorities[:3],
        "field_checks": checks[:4],
        "caution": context["disclaimer"],
        "source": "rule_based",
    }


COACH_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "required": ["summary", "decision", "evidence", "priorities", "field_checks", "caution"],
    "properties": {
        "summary": {"type": "string"},
        "decision": {"type": "string"},
        "evidence": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["label", "value", "interpretation"],
                "properties": {
                    "label": {"type": "string"},
                    "value": {"type": "string"},
                    "interpretation": {"type": "string"},
                },
            },
        },
        "priorities": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "why", "action"],
                "properties": {
                    "title": {"type": "string"},
                    "why": {"type": "string"},
                    "action": {"type": "string"},
                },
            },
        },
        "field_checks": {"type": "array", "items": {"type": "string"}},
        "caution": {"type": "string"},
    },
}


def gemini_response_text(response: dict[str, Any]) -> str:
    """Extract generated text from a Gemini generateContent response."""
    for candidate in response.get("candidates", []):
        for part in candidate.get("content", {}).get("parts", []):
            if isinstance(part.get("text"), str):
                return part["text"]
    raise ValueError("Gemini 응답에 텍스트가 없습니다.")


def gemini_error_notice(error: Exception) -> str:
    """Return a debuggable Gemini failure notice without exposing request URLs or API keys."""
    if isinstance(error, urllib.error.HTTPError):
        hints = {
            400: "요청 형식 또는 응답 스키마를 확인하세요.",
            401: "GEMINI_API_KEY가 올바른지 확인하세요.",
            403: "API 키 권한, API 활성화 상태 또는 프로젝트 제한을 확인하세요.",
            404: "GEMINI_MODEL 이름과 해당 모델의 사용 가능 여부를 확인하세요.",
            429: "Gemini API 할당량 또는 요금제 한도에 도달했습니다.",
        }
        detail = ""
        try:
            body = json.loads(error.read().decode("utf-8", errors="replace"))
            message = body.get("error", {}).get("message")
            if isinstance(message, str):
                detail = f" API 메시지: {message[:240]}"
        except (UnicodeDecodeError, json.JSONDecodeError, AttributeError):
            pass
        hint = hints.get(error.code, "Gemini API 또는 네트워크 상태를 확인하세요.")
        return f"Gemini API 호출 실패 (HTTP {error.code}): {hint}{detail}"
    if isinstance(error, urllib.error.URLError):
        return "Gemini API 네트워크 연결에 실패했습니다. 인터넷 연결과 방화벽 설정을 확인하세요."
    if isinstance(error, TimeoutError):
        return "Gemini API 응답 시간이 초과되었습니다. 잠시 후 다시 시도하세요."
    if isinstance(error, json.JSONDecodeError):
        return "Gemini API 응답을 JSON으로 해석하지 못했습니다. 모델 또는 응답 형식을 확인하세요."
    return "Gemini API 응답이 예상한 코치 형식과 다릅니다. 모델과 응답 스키마를 확인하세요."


def valid_coach(payload: Any) -> bool:
    if not isinstance(payload, dict):
        return False
    if not all(isinstance(payload.get(name), str) for name in ["summary", "decision", "caution"]):
        return False
    return all(isinstance(payload.get(name), list) for name in ["evidence", "priorities", "field_checks"])


def ollama_host() -> str:
    """Return a local-only Ollama host to avoid treating a local fallback as a remote service."""
    host = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
    parsed = urlparse(host)
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("OLLAMA_HOST는 로컬 http://127.0.0.1 또는 localhost 주소여야 합니다.")
    return host


def get_ollama_model(host: str) -> str | None:
    """Choose an installed local model without downloading or exposing anything."""
    request = urllib.request.Request(f"{host}/api/tags", method="GET")
    with urllib.request.urlopen(request, timeout=1.0) as result:
        tags = json.load(result)
    models = [item.get("name") for item in tags.get("models", []) if isinstance(item.get("name"), str)]
    configured = os.environ.get("OLLAMA_MODEL")
    if configured:
        return configured if configured in models else None
    return models[0] if models else None


def ollama_coach(context: dict[str, Any]) -> tuple[dict[str, Any] | None, str | None]:
    """Use a local Ollama model if it is running and has an installed model."""
    try:
        host = ollama_host()
        model = get_ollama_model(host)
        if not model:
            configured = os.environ.get("OLLAMA_MODEL")
            message = f"로컬 모델 '{configured}'을 찾을 수 없습니다." if configured else "Ollama에 설치된 로컬 모델이 없습니다."
            return None, message
        system = (
            "You are a cautious Korean startup-market coach. Return Korean only. "
            "Use only supplied JSON facts and calculations. Never invent data, laws, competitors, "
            "or individual-store outcomes. Do not claim a guarantee or individual survival probability. "
            "Distinguish the market ML estimate from the user-input scenario."
        )
        request_body = {
            "model": model,
            "system": system,
            "prompt": json.dumps(context, ensure_ascii=False),
            "format": COACH_SCHEMA,
            "stream": False,
            "options": {"temperature": 0.2},
        }
        request = urllib.request.Request(
            f"{host}/api/generate",
            data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=45) as result:
            parsed = json.load(result)
        coach = json.loads(parsed["response"])
        if not valid_coach(coach):
            raise ValueError("로컬 모델 응답이 기대한 구조와 다릅니다.")
        coach["source"] = "ollama"
        coach["model"] = model
        return coach, None
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, KeyError, json.JSONDecodeError) as error:
        return None, f"Ollama를 사용할 수 없습니다: {type(error).__name__}"


def gemini_coach(context: dict[str, Any]) -> dict[str, Any]:
    """Request a structured Gemini interpretation without exposing keys to the browser."""
    api_key = os.environ.get("GEMINI_API_KEY")
    fallback = rule_based_coach(context)
    if not api_key:
        fallback["notice"] = "GEMINI_API_KEY가 없어 규칙 기반 코치로 안내합니다."
        return fallback
    model = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash")
    instructions = (
        "You are a cautious Korean startup-market coach. Return Korean only. "
        "Use only the supplied JSON facts and calculations. Never invent data, laws, competitors, "
        "or claims about an individual store. Do not call the result a guarantee or an individual "
        "survival probability. Clearly distinguish the market ML estimate from the user-input scenario."
    )
    request_body = {
        "systemInstruction": {"parts": [{"text": instructions}]},
        "contents": [{"role": "user", "parts": [{"text": json.dumps(context, ensure_ascii=False)}]}],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
            "responseJsonSchema": COACH_SCHEMA,
        },
    }
    request = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{quote(model, safe='-._')}:generateContent?key={quote(api_key, safe='')}",
        data=json.dumps(request_body, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as result:
            parsed = json.load(result)
        coach = json.loads(gemini_response_text(parsed))
        if not valid_coach(coach):
            raise ValueError("The GenAI response did not match the expected structure.")
        coach["source"] = "gemini"
        coach["model"] = model
        return coach
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, ValueError, json.JSONDecodeError) as error:
        fallback["notice"] = gemini_error_notice(error)
        return fallback


def generate_coach(context: dict[str, Any]) -> dict[str, Any]:
    """Select local Ollama, then optional Gemini, then the deterministic fallback."""
    provider = os.environ.get("GENAI_PROVIDER", "auto").lower()
    # Vercel Functions run in a remote serverless environment, so their
    # localhost is not the visitor's PC and cannot host the user's Ollama.
    if os.environ.get("VERCEL") and provider == "auto":
        provider = "gemini" if os.environ.get("GEMINI_API_KEY") else "rules"
    fallback = rule_based_coach(context)
    if provider not in {"auto", "ollama", "gemini", "rules"}:
        fallback["notice"] = "GENAI_PROVIDER 값이 올바르지 않아 규칙 기반 코치로 안내합니다."
        return fallback
    if provider == "rules":
        fallback["notice"] = "GENAI_PROVIDER=rules 설정에 따라 규칙 기반 코치로 안내합니다."
        return fallback
    if provider in {"auto", "ollama"}:
        coach, issue = ollama_coach(context)
        if coach:
            return coach
        if provider == "ollama":
            fallback["notice"] = issue or "Ollama 로컬 모델을 사용할 수 없습니다."
            return fallback
    if provider in {"auto", "gemini"} and os.environ.get("GEMINI_API_KEY"):
        return gemini_coach(context)
    if provider == "gemini":
        fallback["notice"] = "GEMINI_API_KEY가 없어 규칙 기반 코치로 안내합니다."
    else:
        fallback["notice"] = "Ollama 로컬 모델과 Gemini API 키가 없어 규칙 기반 코치로 안내합니다."
    return fallback


class DashboardHandler(SimpleHTTPRequestHandler):
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

    def do_POST(self) -> None:  # noqa: N802 - required by BaseHTTPRequestHandler
        endpoint = urlparse(self.path).path
        if endpoint not in {"/api/scenario", "/api/coach"}:
            self.json_response({"error": "요청 경로를 찾을 수 없습니다."}, HTTPStatus.NOT_FOUND)
            return
        try:
            context = scenario_payload(self.read_json())
            if endpoint == "/api/scenario":
                self.json_response(context)
            else:
                self.json_response(generate_coach(context))
        except json.JSONDecodeError:
            self.json_response({"error": "JSON 요청 형식이 올바르지 않습니다."}, HTTPStatus.BAD_REQUEST)
        except ValueError as error:
            self.json_response({"error": str(error)}, HTTPStatus.BAD_REQUEST)
        except Exception:
            self.json_response({"error": "분석을 처리하지 못했습니다. 입력값을 다시 확인하세요."}, HTTPStatus.INTERNAL_SERVER_ERROR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the OpenSafe AI dashboard.")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--open", action="store_true", help="Open the dashboard in the default browser")
    args = parser.parse_args()

    os.chdir(ROOT)
    address = f"http://127.0.0.1:{args.port}"
    server = ThreadingHTTPServer(("127.0.0.1", args.port), DashboardHandler)
    print(f"OpenSafe AI dashboard: {address}")
    print("Press Ctrl+C to stop the server.")
    if args.open:
        webbrowser.open(address)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nDashboard server stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
