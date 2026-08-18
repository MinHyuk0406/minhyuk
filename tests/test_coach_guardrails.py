"""Deterministic evaluation cases for GenAI coach output constraints."""

from __future__ import annotations

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import valid_coach


def valid_payload() -> dict:
    return {
        "summary": "동일 업종 후보지 비교를 위한 사전 검토 결과입니다.",
        "decision": "계약 전 현장 수요와 비용 조건을 추가로 확인하세요.",
        "caution": "개별 점포의 생존이나 수익을 판단하는 결과가 아닙니다.",
        "evidence": [
            {
                "label": "AI 다음 분기 폐업률",
                "value": "3.2%",
                "interpretation": "행정동·업종 단위의 상대 위험 신호입니다.",
            },
            {
                "label": "기본 시나리오 월 손익",
                "value": "120만 원",
                "interpretation": "사용자가 입력한 매출과 비용 가정의 계산 결과입니다.",
            },
        ],
        "priorities": [
            {
                "title": "현장 관측",
                "why": "평균 지표가 실제 후보 점포에도 적용되는지 확인해야 합니다.",
                "action": "평일과 주말의 고객 수를 각각 관찰하세요.",
            }
        ],
        "field_checks": ["경쟁점의 가격과 영업시간을 확인하세요.", "임대료 외 관리비를 고정비에 반영하세요."],
    }


class CoachGuardrailTests(unittest.TestCase):
    def test_grounded_structure_is_accepted(self) -> None:
        self.assertTrue(valid_coach(valid_payload()))

    def test_certainty_claim_is_rejected(self) -> None:
        payload = valid_payload()
        payload["summary"] = "이 후보지는 반드시 성공을 보장합니다."
        self.assertFalse(valid_coach(payload))

    def test_unapproved_evidence_label_is_rejected(self) -> None:
        payload = valid_payload()
        payload["evidence"][0]["label"] = "주변 경쟁점 수"
        self.assertFalse(valid_coach(payload))


if __name__ == "__main__":
    unittest.main()
