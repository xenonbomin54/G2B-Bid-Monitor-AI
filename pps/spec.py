# -*- coding: utf-8 -*-
"""규격서·과업지시서에서 특정 모델명 후보를 뽑는다 (v9).

dev 측정: v9 정답 근거 2건이 첨부 문서 61%·71% 지점에 있어 섹션 파서가 놓쳤다
(입력누락). 모델명은 한국어 신호어 없이 "Chipset: GB10 Grace Blackwell Superchip"
처럼 영문 제품코드로만 등장하기도 한다. 그래서 문서 전체를 훑어
① 제조사·모델명 표기 줄 ② 영문+숫자 제품코드가 있는 줄 을 후보로 모아
G3 프롬프트에 별도 블록으로 넣는다. 판정(동등이상 허용 문구 유무 등)은 LLM 이 한다.
"""
from __future__ import annotations

import re
from typing import List

from .records import Record

# ① 명시적 표기
_LABEL = re.compile(
    r"(제조사|제조업체|모델명|모델\s*[:：]|모델번호|제품명\s*[:：]|품명\s*[:：]|"
    r"브랜드|상표|Model|Brand|Manufacturer|Chipset|CPU|GPU|Processor)", re.I)

# ② 영문 제품코드: 대문자 시작 + 숫자 포함 토큰 (GB10, RTX4090, M4E/T, X-500)
_CODE = re.compile(r"\b[A-Z][A-Za-z]{0,10}[-/ ]?\d{1,5}[A-Za-z0-9/-]{0,8}\b")

# ③ 대체 허용 문구 — 있으면 위반이 아닐 가능성이 높다 (모델에게 힌트)
_EQUIV = re.compile(r"동등\s*(이상|한|급)|동급|이상의?\s*(제품|성능|사양)|또는\s*이와\s*(동등|유사)|equivalent", re.I)

# 제외: 법령·문서 참조 번호, 날짜, 전화 등
_NOISE = re.compile(r"제\s*\d+\s*조|\d{4}[.\-]\d{1,2}|G2B|KS\s?[A-Z]?\s?\d|ISO\s?\d|IEC|KC\b|Ver\.?|V\d\.\d")


def model_name_candidates(rec: Record, limit: int = 25, width: int = 110) -> List[str]:
    """첨부 문서 우선, 없으면 공고문에서 모델명 후보 줄을 모은다."""
    text = rec.text_of("규격서", "과업지시서", "제안요청서") or rec.notice_text
    out: List[str] = []
    seen = set()
    for line in text.split("\n"):
        s = line.strip()
        if len(s) < 4 or _NOISE.search(s):
            continue
        hit = _LABEL.search(s) or _CODE.search(s)
        if not hit:
            continue
        key = re.sub(r"\s+", " ", s)[:80]
        if key in seen:
            continue
        seen.add(key)
        out.append(re.sub(r"\s+", " ", s)[:width])
        if len(out) >= limit:
            break
    return out


def equivalence_note(rec: Record) -> str:
    text = rec.text_of("규격서", "과업지시서", "제안요청서") or rec.notice_text
    m = _EQUIV.search(text)
    if not m:
        return "대체 허용 문구('동등 이상' 등) 없음"
    s, e = max(0, m.start() - 40), min(len(text), m.end() + 40)
    return "대체 허용 문구 있음: …" + re.sub(r"\s+", " ", text[s:e]) + "…"


def candidate_block(rec: Record) -> str:
    cands = model_name_candidates(rec)
    if not cands:
        return "모델명·제조사 표기로 보이는 줄을 찾지 못했다."
    lines = [f"- {c}" for c in cands]
    return "\n".join(lines) + "\n" + equivalence_note(rec)
