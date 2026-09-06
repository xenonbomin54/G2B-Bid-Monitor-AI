# -*- coding: utf-8 -*-
"""중기부고시 경쟁제품 대조 — 이 공고의 조달 대상이 중기간 경쟁제품인가.

v10·v11·v13 의 전제조건이고, v12(일반제품에 직생 요구)는 그 반대 조건이다.
네 항목 = 점수의 16.7% 가 이 판별 하나에 걸려 있다.

근거 자료
  data/법령패키지/중기부고시/중기부고시_경쟁제품_세부품명.csv (616개 세부품명)
  = 중소벤처기업부 고시 제2025-96호(2025.8.29.) 경쟁제품 지정 내역

dev 실측으로 확인한 것
  · meta.세부품명번호목록 이 등록된 공고는 200건 중 78건뿐이고,
    v10~v13 정답 양성 25건 중 meta 로 경쟁제품이 잡히는 건은 **0건**이다.
    → 메타 필드 단독으로는 판별 불가.
  · 본문에 10자리 세부품명번호가 적힌 경우가 있어 부분적으로 잡힌다.
  → 번호·품명 양쪽으로 후보를 모으고, 최종 판단은 LLM 에 맡긴다.
    이 모듈은 **판정하지 않고 후보를 제시**한다.
"""
from __future__ import annotations

import csv
import io
import os
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set

from .records import Record, nfc

_NUM = re.compile(r"\d{10}")


@dataclass
class GosiItem:
    세부품명번호: str
    세부품명: str
    제품명: str
    대분류: str
    특이사항: str = ""
    공사용자재직접구매: bool = False


@dataclass
class Gosi:
    by_num: Dict[str, GosiItem] = field(default_factory=dict)
    by_name: Dict[str, GosiItem] = field(default_factory=dict)

    @property
    def numbers(self) -> Set[str]:
        return set(self.by_num)

    def names_sorted(self) -> List[str]:
        """긴 이름부터 — 짧은 이름이 긴 이름의 부분문자열일 때 오탐을 줄인다."""
        return sorted(self.by_name, key=len, reverse=True)


_CACHE: Dict[str, Gosi] = {}


def load_gosi(data_dir: str) -> Gosi:
    """중기부고시 세부품명 CSV 로드 (BOM 있음 주의)."""
    path = os.path.join(data_dir, "법령패키지", "중기부고시",
                        "중기부고시_경쟁제품_세부품명.csv")
    if path in _CACHE:
        return _CACHE[path]

    g = Gosi()
    with io.open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            num = (row.get("세부품명번호") or "").strip()
            name = nfc((row.get("세부품명") or "").strip())
            if not num or not name:
                continue
            item = GosiItem(
                세부품명번호=num,
                세부품명=name,
                제품명=nfc((row.get("제품명") or "").strip()),
                대분류=nfc((row.get("대분류") or "").strip()),
                특이사항=nfc((row.get("특이사항") or "").strip()),
                공사용자재직접구매=bool((row.get("공사용자재직접구매") or "").strip()),
            )
            g.by_num[num] = item
            g.by_name.setdefault(name, item)
    _CACHE[path] = g
    return g


# 품명 대조에서 잡음을 만드는 일반어 — 단독으로는 근거로 쓰지 않는다.
_STOP = {
    "기타", "그밖의것", "부품", "제품", "장치", "기기", "용품", "자재", "재료",
    "공사", "용역", "서비스", "시스템", "설비", "장비",
}


@dataclass
class Match:
    번호: List[str] = field(default_factory=list)       # 본문/메타에서 찾은 고시 번호
    품명: List[str] = field(default_factory=list)       # 본문에서 찾은 고시 세부품명
    출처: Dict[str, str] = field(default_factory=dict)  # 품명 → 주변 원문 스니펫

    @property
    def any(self) -> bool:
        return bool(self.번호 or self.품명)

    def summary(self, limit: int = 8) -> str:
        parts = []
        if self.번호:
            parts.append("번호 " + ", ".join(self.번호[:limit]))
        if self.품명:
            parts.append("품명 " + ", ".join(self.품명[:limit]))
        return " / ".join(parts) if parts else "없음"


def match_record(rec: Record, gosi: Gosi, name_min_len: int = 3) -> Match:
    """공고에서 중기부고시 경쟁제품 흔적을 찾는다. 판정이 아니라 후보 수집."""
    text = rec.full_text
    m = Match()

    # 1) 세부품명번호 (메타 + 본문). 정확한 신호라 우선한다.
    nums = set(rec.세부품명번호) | set(_NUM.findall(text))
    m.번호 = sorted(n for n in nums if n in gosi.by_num)

    # 2) 세부품명 문자열. 짧거나 일반적인 이름은 제외한다.
    for name in gosi.names_sorted():
        if len(name) < name_min_len or name in _STOP:
            continue
        i = text.find(name)
        if i < 0:
            continue
        m.품명.append(name)
        s, e = max(0, i - 40), min(len(text), i + len(name) + 40)
        m.출처[name] = re.sub(r"\s+", " ", text[s:e]).strip()
        if len(m.품명) >= 20:
            break
    return m


def candidate_lines(rec: Record, gosi: Gosi, limit: int = 6) -> str:
    """LLM 프롬프트에 넣을 후보 목록 문자열.

    모델에게 '이 공고의 조달 대상이 경쟁제품인가'를 물을 때 근거로 함께 준다.
    후보가 없으면 그 사실도 알려야 한다 — 없음을 '판단 불가'로 오해하면 안 된다.
    """
    m = match_record(rec, gosi)
    if not m.any:
        return "중기부고시 경쟁제품 목록과 일치하는 품명·품명번호를 찾지 못했다."
    lines = []
    for n in m.번호[:limit]:
        it = gosi.by_num[n]
        note = f" · 특이사항: {it.특이사항}" if it.특이사항 else ""
        lines.append(f"- [번호일치] {n} {it.세부품명} (제품명 {it.제품명}){note}")
    for name in m.품명[:limit]:
        it = gosi.by_name[name]
        note = f" · 특이사항: {it.특이사항}" if it.특이사항 else ""
        lines.append(f"- [품명일치] {it.세부품명번호} {name}{note}"
                     f"\n    원문: …{m.출처.get(name, '')}…")
    return "\n".join(lines)
