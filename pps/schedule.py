# -*- coding: utf-8 -*-
"""일정 추출·검산 — 설명회일·제안서 제출마감일과 법정 기간 (v22·v23).

근거 조문
  「지방자치단체 입찰시 낙찰자 결정기준」 제7장 제3절 2.다
    제안요청서 설명은 **제안서 제출마감일의 전일부터 기산하여** 아래 기간 전에 실시해야 한다.
      추정가격 10억원 이상 40일 / 10억 미만 1억 이상 20일 / 1억원 미만 10일
    이 경우 **입찰공고는 설명일의 전일부터 기산하여 7일 전**에 공고해야 한다.
  지방계약법 시행령 제35조⑤ (입찰공고 자체의 시기, 같은 10/20/40일 구간)

dev 검증에서 확인한 것
  v23 양성은 '입찰공고가 짧은 것'이 아니라 **설명회를 늦게 잡아 설명회 이후 준비기간이
  부족한 것**이다. 공고게시일 기준으로만 재면 정반대로 상관한다(TP0 FP6 FN5).
    PPS-DEV-27  공고 01-14 → 설명회 02-03 → 제출 02-06 : 설명회 후 3일 (필요 10일) 위반
    PPS-DEV-139 공고 03-11 → 설명회 03-18 → 개찰 03-23 : 설명회 후 5일 (필요 10일) 위반
"""
from __future__ import annotations

import datetime
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .records import Record

억 = 100_000_000

# 2026. 2. 3.(화) / 2026.02.19. / 2026-03-11 / 2026년 3월 18일
_DATE = re.compile(
    r"(20\d{2})\s*[.\-년]\s*(\d{1,2})\s*[.\-월]\s*(\d{1,2})\s*[.일]?")
# 03.11. 처럼 연도가 생략된 경우 (앞선 연도를 물려받는다)
_MD = re.compile(r"(?<![\d.])(\d{1,2})\s*[.]\s*(\d{1,2})\s*[.]")

설명회_KW = re.compile(
    r"(제안요청서?\s*설명|제안설명|사업\s*설명회|현장\s*설명회|현장설명|설명회)")
마감_KW = re.compile(
    r"(제안서[^\n]{0,12}(제출|접수)|입찰서[^\n]{0,12}제출|제출\s*마감|접수\s*마감|"
    r"가격입찰서[^\n]{0,12}제출)")


def _mk(y: int, m: int, d: int) -> Optional[datetime.date]:
    try:
        return datetime.date(y, m, d)
    except ValueError:
        return None


def meta_date(v) -> Optional[datetime.date]:
    s = str(v or "")
    if len(s) != 8 or not s.isdigit():
        return None
    return _mk(int(s[:4]), int(s[4:6]), int(s[6:8]))


def dates_near(text: str, kw: re.Pattern, window: int = 220,
               year_hint: Optional[int] = None) -> List[datetime.date]:
    """키워드 주변에서 날짜를 뽑는다. 키워드 뒤쪽을 우선한다."""
    out: List[datetime.date] = []
    for m in kw.finditer(text):
        seg = text[m.end(): m.end() + window]
        for dm in _DATE.finditer(seg):
            d = _mk(int(dm.group(1)), int(dm.group(2)), int(dm.group(3)))
            if d:
                out.append(d)
        if not out and year_hint:
            for dm in _MD.finditer(seg):
                d = _mk(year_hint, int(dm.group(1)), int(dm.group(2)))
                if d:
                    out.append(d)
        if out:
            break
    return out


@dataclass
class Schedule:
    공고일: Optional[datetime.date]
    개찰일: Optional[datetime.date]
    설명회일: Optional[datetime.date]
    마감일: Optional[datetime.date]
    설명회_있음: bool

    @property
    def 실질마감(self) -> Optional[datetime.date]:
        """제안서 제출마감일. 본문에서 못 찾으면 개찰예정일자로 대신한다."""
        return self.마감일 or self.개찰일


def extract(rec: Record) -> Schedule:
    t = rec.notice_text
    공고 = meta_date(rec.meta.get("공고게시일자"))
    개찰 = meta_date(rec.meta.get("개찰예정일자"))
    hint = 공고.year if 공고 else None

    설명 = dates_near(t, 설명회_KW, year_hint=hint)
    마감 = dates_near(t, 마감_KW, year_hint=hint)

    # 공고일 이전 날짜는 일정이 아니다(실적 기준일 등) — 걸러낸다
    def pick(cands: List[datetime.date]) -> Optional[datetime.date]:
        ok = [d for d in cands if not 공고 or d >= 공고]
        return min(ok) if ok else None

    return Schedule(
        공고일=공고, 개찰일=개찰,
        설명회일=pick(설명), 마감일=pick(마감),
        설명회_있음=bool(설명회_KW.search(t)),
    )


def 법정_설명회_선행일수(추정가격: int) -> int:
    """설명회는 제안서 제출마감일 기준 며칠 전에 열려야 하는가."""
    if 추정가격 >= 10 * 억:
        return 40
    if 추정가격 >= 1 * 억:
        return 20
    return 10


def check_v23(rec: Record) -> Tuple[Optional[bool], str]:
    """(위반 판단, 설명문). 판단 불가면 (None, 설명).

    규칙 판정을 강제하지 않고 계산 결과를 사실로 돌려준다 —
    dev 양성이 5건뿐이라 규칙을 못 박으면 과적합한다. 최종 판단은 LLM 이 한다.
    """
    if not (rec.is_지방 and rec.is_협상):
        return False, "협상에 의한 계약 + 지방계약법 이 아니므로 v23 성립 불가."

    s = extract(rec)
    if not s.설명회_있음:
        return False, "설명회 개최 문구가 없다 → v23 성립 불가."

    need = 법정_설명회_선행일수(rec.추정가격)
    lines = [f"설명회 개최 문구 있음. 법정 요건: 설명회는 제안서 제출마감일 "
             f"{need}일 전까지 실시(추정가격 {rec.추정가격:,}원 기준)."]

    if s.설명회일 and s.실질마감:
        gap = (s.실질마감 - s.설명회일).days
        src = "제안서 제출마감일" if s.마감일 else "개찰예정일자(마감일 대용)"
        lines.append(f"본문 설명회일 {s.설명회일} → {src} {s.실질마감} = {gap}일")
        if gap <= 0:
            # 설명회가 마감일과 같거나 그 뒤면 일정이 성립하지 않는다.
            # 위반이 아니라 날짜 추출이 어긋난 것이다 — dev 거짓양성 7건 중 6건이 이 유형.
            lines.append("→ 설명회가 마감일 이후이거나 같다. 날짜 추출이 어긋난 것으로 보이므로 "
                         "계산을 신뢰하지 말고 문서를 직접 읽어 판단하라.")
            return None, " ".join(lines)
        if gap < need:
            lines.append(f"→ {need}일에 {need - gap}일 모자람. **v23 위반에 해당**")
            return True, " ".join(lines)
        lines.append(f"→ {need}일 이상 확보됨. 이 요건은 충족.")
        if s.공고일 and s.설명회일:
            g2 = (s.설명회일 - s.공고일).days
            lines.append(f"공고일 {s.공고일} → 설명회 {g2}일 "
                         f"(법정 7일 {'충족' if g2 >= 7 else '미달 — 위반'})")
            if g2 < 7:
                return True, " ".join(lines)
        return False, " ".join(lines)

    lines.append("본문에서 설명회일 또는 제출마감일을 특정하지 못했다 — 문서를 직접 읽고 판단하라.")
    return None, " ".join(lines)


def fact_line(rec: Record) -> str:
    v, msg = check_v23(rec)
    tag = {True: "위반 소지", False: "요건 충족", None: "판단 보류"}[v]
    return f"[v23 일정 검산 · {tag}] {msg}"
