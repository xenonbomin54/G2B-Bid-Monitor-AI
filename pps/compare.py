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

from . import law
from .records import Record

# --------------------------------------------------------------------------- 금액

# 금액 표기가 제각각이다. dev200 에서 '추정금액 : ￦360,400,000' 를 놓쳐
# v24 억제 규칙이 발동하지 않았다(거짓양성 20건 중 다수가 이 유형).
#   금120,000,000원 / ￦360,400,000 / 1,234,000 원 / 61560000원
_AMT = re.compile(
    r"(?:[￦₩]\s*|금\s*)?(\d{1,3}(?:,\d{3})+|\d{6,})\s*(?:원|$|[^\d,])")
_AMT_UNIT = re.compile(r"(\d+(?:\.\d+)?)\s*(억|천만|백만|만)\s*원")
_UNIT = {"억": 100_000_000, "천만": 10_000_000, "백만": 1_000_000, "만": 10_000}

# ⚠️ 자릿수 단위가 뒤에 붙는 표기 — "사업예산: 1,250,000천원" (= 12.5억).
#    이걸 원으로 읽으면 1,250,000 원이 되어 1,000배 틀린다. dev 에서 PPS-DEV-25 가
#    이 때문에 v24 거짓양성으로 잡혔다. _AMT 보다 **먼저** 처리하고 그 구간은 건너뛴다.
_AMT_SUFFIX = re.compile(
    r"(?:[￦₩]\s*|금\s*)?(\d{1,3}(?:,\d{3})+|\d+)\s*(천원|백만원|천만원|억원|만원)")
_SUFFIX_UNIT = {"천원": 1_000, "만원": 10_000, "백만원": 1_000_000,
                "천만원": 10_000_000, "억원": 100_000_000}


def amounts_in(text: str) -> List[int]:
    out: List[int] = []
    # 단위 접미사가 붙은 표기를 먼저 잡고, 그 구간은 뒤 패턴에서 제외한다.
    spans = []
    for m in _AMT_SUFFIX.finditer(text):
        out.append(int(float(m.group(1).replace(",", "")) * _SUFFIX_UNIT[m.group(2)]))
        spans.append((m.start(), m.end()))

    def covered(i: int) -> bool:
        return any(s <= i < e for s, e in spans)

    for m in _AMT.finditer(text):
        if covered(m.start()):
            continue
        out.append(int(m.group(1).replace(",", "")))
    for m in _AMT_UNIT.finditer(text):
        if covered(m.start()):
            continue
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


# --------------------------------------------------------------------------- 공동수급 지분율 (v21)

# "최소 지분율 3% 이상" / "지분율은 100분의 5 이상" / "출자비율 5%" / "지분참여 비율 10% 이상"
_지분율 = re.compile(
    r"(?:최소\s*)?(?:지분\s*율|지분\s*비율|지분\s*참여\s*비율|출자\s*비율|지분)"
    r"\s*(?:은|는|이|가|을|를)?\s*[^\n%]{0,30}?"
    r"(?:(\d{1,3}(?:\.\d+)?)\s*%|100\s*분의\s*(\d{1,3}))")
_공동 = re.compile(r"공동\s*(?:수급|계약|도급|이행)|공동수급체|공동협정서")
_분담이행 = re.compile(r"분담\s*이행")
# 분담이행방식에는 최소지분율이 적용되지 않는다(지방 집행기준 제6장 제2절 1-나-3).
# 다만 두 방식을 함께 허용하는 공고가 흔하므로, '분담이행' 표기만으로 보류하면
# 공동이행 쪽 위반을 놓친다 → 공동이행 표기가 함께 있으면 보류하지 않는다.
_공동이행 = re.compile(r"공동\s*이행")
# 지분율 표기가 '공동수급체 구성원의 최소지분율'을 말하는지 가르는 주변 단서.
# 이것 없이 숫자만 뽑으면 '부가세 10%'·'지분율 평가 배점' 같은 것이 섞인다.
_공동문맥 = re.compile(r"공동|구성원|수급체|최소\s*지분|대표자|출자")


def joint_shares(text: str, window: int = 120) -> List[Tuple[str, float]]:
    """본문에서 (인용구, 백분율) 목록. 판단하지 않고 후보만 모은다.

    주변 window 자 안에 공동수급 문맥 단서가 있는 것만 남긴다 — **후보 좁히기**다.
    좁힌 뒤에도 '지분율 평가 배점 10%' 같은 것이 남을 수 있으므로
    값만 주지 않고 **인용구를 함께** 준다. 판단은 모델이 한다.
    (STATUS.md 원칙 3: 정규식으로 의미 매칭을 흉내내지 않는다.)
    """
    out: List[Tuple[str, float]] = []
    seen = set()
    for m in _지분율.finditer(text):
        raw = m.group(1) or m.group(2)
        if raw is None:
            continue
        try:
            pct = float(raw)
        except ValueError:
            continue
        if not 0 < pct <= 100:
            continue
        ctx = text[max(0, m.start() - window): m.end() + window]
        if not _공동문맥.search(ctx):
            continue
        quote = re.sub(r"\s+", " ",
                       text[max(0, m.start() - 60): m.end() + 30]).strip()
        key = quote[:40]
        if key in seen:
            continue
        seen.add(key)
        out.append((quote, pct))
    return out


def check_v21(rec: Record) -> Tuple[Optional[bool], Optional[str]]:
    """v21 — 공고가 정한 공동수급체 최소지분율이 법정 기준 미만인지 판정한다.

    반환 (판정, 근거문구). 판정 None = 보류(LLM 에 맡긴다).

    dev 200건 검증 (tools/probe_v21.py)
      TP 6 · FP 0 · FN 0 → **규칙 F1 1.000** vs 같은 항목 LLM F1 0.500.
      규칙이 LLM 보다 확실히 나으므로 강제한다 — v23 과 같은 근거다.
        PPS-DEV-25  지방 3.0% < 5%      PPS-DEV-049 지방 4.0% < 5%
        PPS-DEV-055 국가 5.0% < 10%     PPS-DEV-056 국가 0.5% < 10%
        PPS-DEV-058 국가 5.0% < 10%     PPS-DEV-059 지방 2.0% < 5%
      양성 6건 중 5건은 정답 근거문구가 빈칸이지만, 원문에는 지분율이 또박또박
      적혀 있었다 — 라벨러가 인용을 생략한 것이지 근거가 없는 것이 아니었다.

    보류하는 경우 (규칙이 답하지 않고 LLM 에 넘긴다)
      · 지분율 표기를 못 찾음 — '부재'를 위반으로 보지 않는다. v21 은 부재탐지 항목이 아니다.
      · 분담이행방식 — 조문상 최소지분율이 적용되지 않는다
        (지방 집행기준 제6장 제2절 1-나-3, 국가도 공동이행방식 조항이다).
    """
    t = rec.full_text
    if _분담이행.search(t) and not _공동이행.search(t):
        return None, None

    found = joint_shares(t)
    if not found:
        return None, None

    기준 = law.공동_최소지분율_기준(rec.적용계약법, rec.업무구분, rec.추정가격)
    worst = min(found, key=lambda x: x[1])
    if worst[1] < 기준:
        return True, worst[0]
    return False, None


def joint_share_check(rec: Record) -> str:
    """공동수급체 최소지분율 기준을 계산해 사실로 준다 (v21).

    왜 계산해서 주는가
      모델은 v21 판정 기준("법정 기준보다 낮으면 위반")은 받지만 **기준값 자체를
      모른다.** 지방 5% / 국가 10% 이고 조문에서 계산되는 값이므로 코드가 준다.
      (fact_block 이 금액 구간·지역제한 허용여부를 계산해 주는 것과 같은 이유)

    ⚠️ 20% 가감 단서를 적용하면 안 된다 — law.py 의 주석 참조.
       가감 하한(4%/8%)을 쓰면 PPS-DEV-049(지방 4.0%)를 놓친다.

    판정 자체는 `check_v21` 이 규칙으로 강제한다(dev F1 1.000). 이 블록은
    모델이 근거문구를 고를 수 있게 같은 사실을 프롬프트에도 넣어 주는 것이다.
    """
    기준 = law.공동_최소지분율_기준(rec.적용계약법, rec.업무구분, rec.추정가격)
    parts = [f"법정 최소지분율 {기준:g}% ({rec.적용계약법}). "
             f"공고가 정한 구성원별 최소지분율이 {기준:g}% 미만이면 위반"]

    방식 = str(rec.meta.get("공동도급구성방식") or "").strip()
    if 방식:
        parts.append(f"등록 공동도급구성방식={방식}")

    t = rec.full_text
    if not _공동.search(t) and not 방식:
        parts.append("본문에 공동수급 언급 없음")
    if _분담이행.search(t):
        parts.append("본문에 '분담이행' 표기 있음 — 분담이행방식에는 최소지분율이 적용되지 않는다")

    found = joint_shares(t)
    if not found:
        parts.append("본문에서 지분율 수치를 찾지 못했다(표기가 없거나 형식이 달라서일 수 있다)")
    else:
        parts.append(f"본문에서 찾은 지분율 표기 {len(found)}건")
        for quote, pct in found[:4]:
            flag = f"  ← 기준 {기준:g}% 미만 = 위반" if pct < 기준 else ""
            parts.append(f"  · {pct:g}%{flag}  …{quote[:140]}…")

    return "\n  ".join(parts)


# --------------------------------------------------------------------------- v24 값 대조

_NUM = r"(\d{1,3}(?:,\d{3})+|\d{6,})"
# 레이블이 붙은 금액만 본다. 레이블 없이 본문 금액을 다 긁으면 상투 문구에 걸린다
# (STATUS.md: 단순 규칙 정밀도 0.12 로 실패한 원인).
_LBL_추정가 = re.compile(r"추정\s*가격\s*[:：]?\s*(?:금\s*)?" + _NUM)
_LBL_예산 = re.compile(
    r"(?:배정\s*예산|사업\s*예산|예산\s*액|총\s*사업비)\s*[:：]?\s*(?:금\s*)?" + _NUM)
# "입찰방법 : 제한경쟁입찰" / "계약방법 : 일반경쟁(총액…)" — 레이블 뒤 60자만 본다
_LBL_방법 = re.compile(r"(?:입찰|계약)\s*방(?:법|식)\s*[:：]\s*([^\n]{0,60})")
_경쟁유형 = re.compile(r"(일반경쟁|제한경쟁|지명경쟁|수의계약)")
# 숫자 바로 뒤에 붙는 자릿수 단위 — 붙어 있으면 그 숫자는 원 단위가 아니다.
_SUFFIX_AFTER = re.compile(r"\s*(?:천원|만원|백만원|천만원|억원)")


def check_v24_positive(rec: Record) -> Tuple[bool, Optional[str]]:
    """v24 — 공고문과 나라장터 등록값이 **실제로 다른 값**인 경우만 True.

    ⛔ **파이프라인에 배선하지 않았다.** 측정 기록으로 남긴 것이다.
       dev 200 실측:
         LLM 단독  TP 2 FP 12 FN 6 → F1 0.182
         규칙 단독  TP 2 FP  2 FN 6 → F1 0.333   ← 규칙이 더 낫다
         합집합    TP 2 FP 15 FN 6 → F1 0.160   ← **LLM 보다 나쁘다**
       규칙이 맞춘 2건이 LLM 이 이미 맞춘 것과 **같은 레코드**여서, 양성 전용으로
       합쳐도 새 TP 는 0 이고 FP 만 늘어난다. 규칙 단독으로 넘기면 dev 는 오르지만
       양성 8건 표본이라 09-07(세부품명 게이팅, 양성 19건 → 리더보드 −0.105)과 같은
       도박이 된다. 리더보드로 검증할 여력이 생기면 규칙 단독을 시험해 볼 만하다.

    잡는 두 가지
      1) 레이블 붙은 금액 불일치 — PPS-DEV-29
         본문 "추정가격 34,481,818원" vs 등록 36,118,183원
      2) 레이블 붙은 경쟁유형 불일치 — PPS-DEV-057
         본문 "입찰방법 : 제한경쟁입찰(…)" vs 등록 계약방법=일반경쟁

    수의계약을 경쟁유형 대조에서 제외하는 이유: 등록이 수의계약인 공고가 본문에
    제한경쟁 상투 문구를 달고 있어 거짓양성 2건이 나왔다(PPS-DEV-148 · 194).
    """
    t = rec.notice_text

    def ok(v: int) -> bool:
        return (_close(v, rec.추정가격) or _close(v, rec.배정예산)
                or _close(v, round(rec.추정가격 * 1.1)))

    for pat in (_LBL_추정가, _LBL_예산):
        for m in pat.finditer(t):
            # "사업예산: 1,250,000천원" 처럼 자릿수 단위가 뒤에 붙으면 이 숫자는
            # 원 단위가 아니다 — 그대로 비교하면 1,000배 틀린다(PPS-DEV-25 거짓양성).
            if _SUFFIX_AFTER.match(t, m.end()):
                continue
            v = int(m.group(1).replace(",", ""))
            if not ok(v):
                return True, re.sub(r"\s+", " ",
                                    t[max(0, m.start() - 30): m.end() + 15]).strip()

    reg = (rec.계약방법 or "").replace(" ", "")
    if reg in ("일반경쟁", "제한경쟁"):
        found = set()
        first = None
        for m in _LBL_방법.finditer(t):
            for tm in _경쟁유형.finditer(m.group(1)):
                found.add(tm.group(1))
                if first is None:
                    first = re.sub(r"\s+", " ", m.group(0)).strip()
        if found and reg not in found:
            return True, first
    return False, None


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
