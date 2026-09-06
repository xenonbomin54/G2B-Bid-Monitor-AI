# -*- coding: utf-8 -*-
"""공고 본문 ↔ 나라장터 등록값 대조 (v24 보조) + 조항호 해석.

dev 20건 측정에서 v24 거짓양성이 9건 나왔다. 모델이 '사업소요예산 83,100,000원'
같은 줄을 인용하며 위반이라 했는데, 그 값은 등록값과 **일치**했다. 모델은 숫자를
비교하지 않고 금액이 적힌 줄을 보면 위반이라 답한다.
→ 숫자 대조는 코드가 하고, 결과를 사실로 준다. 모델은 계산하지 않는다.

조항호내용은 발주기관이 나라장터에 등록한 "어느 조항호로 제한한다"는 자기 신고다.
조문 문구 그대로라 기계적으로 해석할 수 있고, 특히
"중소벤처기업부장관이 지정 공고한 물품" = 영 제21조①8호 = **중기간 경쟁제품 입찰**
이라는 사실은 v10·v11·v13 판정의 출발점이다. dev 측정에서 모델이 이 연결을 못 했다.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from .records import Record

# --------------------------------------------------------------------------- 금액

_AMT = re.compile(r"(?:금\s*)?(\d{1,3}(?:,\d{3})+|\d{5,})\s*원")
_AMT_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만)\s*원")
_UNIT = {"억": 100_000_000, "천만": 10_000_000, "백만": 1_000_000, "만": 10_000}


def amounts_in(text: str) -> List[int]:
    out = []
    for m in _AMT.finditer(text):
        out.append(int(m.group(1).replace(",", "")))
    for m in _AMT_UNIT.finditer(text):
        out.append(int(float(m.group(1)) * _UNIT[m.group(2)]))
    return out


def _close(a: int, b: int, tol: float = 0.005) -> bool:
    return a > 0 and b > 0 and abs(a - b) <= max(1000, tol * max(a, b))


def amount_check(rec: Record, head_chars: int = 6000) -> Tuple[str, bool]:
    """(설명문, 본문에서 등록 금액을 찾았는가).

    공고문 앞부분의 모든 금액을 뽑아 배정예산·추정가격과 대조한다.
    부가세 포함/제외(÷1.1) 도 함께 본다 — 예산은 부가세 포함, 추정가격은 제외인 경우가 많다.
    """
    amts = amounts_in(rec.notice_text[:head_chars])
    budget, est = rec.배정예산, rec.추정가격
    found_b = any(_close(a, budget) for a in amts)
    found_e = any(_close(a, est) or _close(a, round(est * 1.1)) for a in amts)
    parts = []
    if budget:
        parts.append(f"배정예산 {budget:,}원 → 본문에 {'있음(일치)' if found_b else '동일 금액 없음'}")
    if est:
        parts.append(f"추정가격 {est:,}원 → 본문에 {'있음(일치)' if found_e else '동일 금액 없음'}")
    if amts:
        top = sorted(set(amts), reverse=True)[:5]
        parts.append("본문 주요 금액: " + ", ".join(f"{a:,}원" for a in top))
    return " / ".join(parts) or "본문에 금액 표기 없음", (found_b or found_e)


def amount_is_matching_quote(rec: Record, quote: Optional[str]) -> bool:
    """모델이 v24 근거로 인용한 줄의 금액이 등록값과 일치하면 True (= 위반 아님)."""
    if not quote:
        return False
    amts = amounts_in(quote)
    if not amts:
        return False
    b, e = rec.배정예산, rec.추정가격
    return all(_close(a, b) or _close(a, e) or _close(a, round(e * 1.1)) for a in amts)


# --------------------------------------------------------------------------- 지역

_광역 = ["서울특별시", "부산광역시", "대구광역시", "인천광역시", "광주광역시", "대전광역시",
       "울산광역시", "세종특별자치시", "경기도", "강원특별자치도", "충청북도", "충청남도",
       "전북특별자치도", "전라남도", "경상북도", "경상남도", "제주특별자치도"]
_광역_short = {"서울": "서울특별시", "부산": "부산광역시", "대구": "대구광역시", "인천": "인천광역시",
            "광주": "광주광역시", "대전": "대전광역시", "울산": "울산광역시", "세종": "세종특별자치시",
            "경기": "경기도", "강원": "강원특별자치도", "충북": "충청북도", "충남": "충청남도",
            "전북": "전북특별자치도", "전남": "전라남도", "경북": "경상북도", "경남": "경상남도",
            "제주": "제주특별자치도"}
_TOKEN = re.compile(r"\[(?:등록)?지역:r\d+\|단위=(기초|리|미상|광역)?\|?광역=([^\]|]+)\]")


def regions_in(text: str) -> Tuple[List[str], bool]:
    """(언급된 광역 목록, 기초단위 토큰 존재 여부)."""
    found = []
    for g in _광역:
        if g in text:
            found.append(g)
    for s, full in _광역_short.items():
        if re.search(rf"(?<![가-힣]){s}(?![가-힣])", text) and full not in found:
            found.append(full)
    basic = False
    for m in _TOKEN.finditer(text):
        if m.group(1) == "기초":
            basic = True
        g = m.group(2).strip()
        if g and g != "미상" and g not in found:
            found.append(g)
    return found, basic


# 소재지 제한을 말하는 표현 — 기초단위 토큰이 '참가자격' 문맥에 있는지 보기 위함
_소재지 = re.compile(r"(주된\s*영업소|본점\s*소재지|본사|영업소의?\s*소재지|소재지|법인등기부)")
_기초토큰 = re.compile(r"\[지역:r\d+\|단위=(?:기초|리)")


def basic_unit_near_restriction(rec: Record, window: int = 260) -> Optional[str]:
    """지역제한 문맥에서 시·군·구(기초) 단위 토큰이 쓰였는지. 쓰였으면 그 구간을 돌려준다.

    ⚠️ 이것만으로 v6 을 단정하지 않는다. dev 검증에서 정밀도 0.21 에 그쳤다
    (기초 토큰이 수요기관 주소 등 자격과 무관한 곳에도 흔히 나온다).
    판정 근거가 아니라 **모델이 확인해 볼 단서**로만 제공한다.
    """
    t = rec.full_text
    for m in _기초토큰.finditer(t):
        s, e = max(0, m.start() - window), m.start() + window
        if _소재지.search(t[s:e]):
            return re.sub(r"\s+", " ", t[max(0, m.start() - 90): m.start() + 90]).strip()
    return None


def region_check(rec: Record) -> str:
    meta_region = str(rec.meta.get("제한지역코드목록") or "")
    flag = rec.meta.get("지역제한여부")
    parts = []
    if not meta_region or meta_region == "None":
        parts.append(f"등록 지역제한여부={flag}, 제한지역 미등록")
    else:
        regs, basic = regions_in(meta_region)
        parts.append("등록 제한지역: " + (", ".join(regs) if regs else meta_region))
        if basic:
            parts.append("등록값이 기초(시·군·구) 단위")
    hit = basic_unit_near_restriction(rec)
    if hit:
        parts.append(f"본문 소재지 제한 문맥에 시·군·구 단위 표기가 보임(단서, 확인 필요): …{hit}…")
    return " / ".join(parts)


# --------------------------------------------------------------------------- 조항호 해석

# ⚠️ 조항호 해석은 **어느 조문 경로인지**만 말한다.
#    "무엇을 제한했다"는 서술을 넣으면 안 된다.
#    dev 40건 측정에서 "중소기업 제한 등록" 이라는 문구 때문에 v16(중소기업 제한이
#    '없는 것'이 위반)이 1.00 → 0.00 으로 무너졌다. 모델이 등록값을 보고
#    "제한이 있으니 위반 아님"으로 갔다. v14·v18 도 같이 오염됐다.
#    조항호는 발주기관의 **등록값**일 뿐이고, 공고문에 실제로 그 제한이 적혔는지는
#    본문으로만 확인해야 한다. 둘이 다른 것이 바로 v24 의 정의다.
_조항호_해석: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"중소벤처기업부장관이\s*지정|중소기업청장이\s*지정"),
     "영 제21조①8호(국가)/제20조①8호(지방) 경로 = **중소기업자간 경쟁제품 입찰**. "
     "→ v10·v11·v13 의 적용 대상이다."),
    (re.compile(r"1억원\s*미만.*(소기업|소상공인)"),
     "영 제21조①10호가목 경로(1억 미만 구간)."),
    (re.compile(r"1억원\s*이상.*고시금액\s*미만.*중소기업"),
     "영 제21조①10호나목 경로(1억~고시금액 구간)."),
    (re.compile(r"판로지원법\s*시행령.*중기업"),
     "판로지원법 시행령 기업규모 제한 경로 — 등록 범위에 **중기업이 포함**되어 있다."),
    (re.compile(r"판로지원법\s*시행령"),
     "판로지원법 시행령 기업규모 제한 경로."),
    (re.compile(r"금액\s*미만.*(본점소재지|영업소재지|지역제한)"),
     "영 제21조①6호 경로 = 기준금액 미만 지역제한."),
    (re.compile(r"특수한\s*기술.*용역"),
     "영 제21조①5호 경로 = 특수기술 용역의 실적·기술보유 제한."),
    (re.compile(r"2천만원\s*초과\s*1억원\s*이하"),
     "지방 소액수의(2인 견적) 경로 — 지역·실적 제한 특례가 가능한 구간."),
    (re.compile(r"협상에\s*의한\s*계약"),
     "협상에 의한 계약 — v22·v23 의 적용 대상."),
]

_등록값_주의 = ("(이 값은 발주기관이 나라장터에 **등록**한 것이다. "
            "공고문에 그 제한이 실제로 적혀 있는지는 본문으로 확인하라 — "
            "등록값만 보고 '기재가 있다'고 판단하지 말 것.)")


def interpret_clause(rec: Record) -> str:
    s = str(rec.meta.get("조항호내용") or "")
    if not s or s == "None":
        return ""
    for pat, meaning in _조항호_해석:
        if pat.search(s):
            return meaning + " " + _등록값_주의
    return ""
