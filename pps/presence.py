# -*- coding: utf-8 -*-
"""부재탐지 항목용 전체문서 존재 스캔.

부재탐지 5항목(v10·v11·v16·v18·v20)은 "있어야 할 기재가 없는 것"이 위반이다.
이 판정에는 LLM보다 정규식이 낫다.

  · 정확성 — "문서 어디에도 없음"은 전수 검사 문제다. LLM은 긴 문맥에서
    needle을 놓치고(26B-A4B의 MRCR 44.1%), 놓친 것을 '없다'고 답한다.
    즉 LLM의 실패 방향이 곧 거짓양성이다. 정규식은 놓치지 않는다.
  · 비용 — 전체 문서를 프롬프트에 넣을 필요가 없다. 토큰 0.
  · 절단 안전 — 프롬프트 예산과 무관하게 원문 전체를 본다.

역할 분담
  정규식  "요구 문구가 문서에 있는가?"  (있음 → 위반 아님, 확정)
  LLM     "그 요구가 이 공고에 필요한가?" (경쟁제품 입찰인가 / SW사업인가 / 금액대는 맞는가)

따라서 이 모듈은 위반 여부를 직접 정하지 않는다. `present=True`면 0 확정,
`present=False`면 '적용 대상인지'를 LLM·게이팅이 판단한 뒤 위반으로 확정한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .records import Record


@dataclass(frozen=True)
class Signal:
    """존재 신호 하나. 하나라도 걸리면 '기재 있음'."""
    name: str
    pattern: re.Pattern
    근거: str = ""


def _p(*alts: str) -> re.Pattern:
    """공백·중점·괄호를 유연하게 흡수하는 패턴. 원문 표기가 제각각이라 필요하다."""
    return re.compile("|".join(alts))


# --------------------------------------------------------------------------- v10 직접생산확인
# 중기간 경쟁제품 입찰에서 직접생산확인증명서 소지를 요구하지 않으면 위반.
# 판로지원법 제9조 / 중기간경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역
V10_직생요구 = [
    Signal("직접생산확인증명서", _p(r"직접\s*생산\s*확인\s*(증명서|서류|필증)")),
    Signal("직접생산확인", _p(r"직접\s*생산\s*확인")),
    Signal("직접생산", _p(r"직접\s*생산(?!\s*확인)")),
    Signal("직생", _p(r"(?<![가-힣])직생(?![가-힣])")),
    Signal("공공구매종합정보망", _p(r"smpp\.go\.kr", r"공공구매\s*종합정보망")),
]

# --------------------------------------------------------------------------- v11 중소기업자 제한
# 중기간 경쟁제품 입찰은 중소기업자만 참여 가능하도록 제한해야 한다.
# 판로지원법 제7조제1항 / 국가 영 제21조제1항제8호 · 지방 영 제20조제1항제8호
V11_중소제한 = [
    Signal("중소기업자", _p(r"중소기업자?(?:\s*간)?\s*(?:경쟁|제한|만|으로|에\s*한)",
                          r"중소기업\s*확인서",
                          r"「?중소기업기본법」?\s*제2조")),
    Signal("중기간경쟁", _p(r"중소기업자\s*간\s*경쟁", r"중기간\s*경쟁")),
]

# --------------------------------------------------------------------------- v16 중소기업 제한 (1억~고시금액)
# 국가 영 제21조제1항제10호나목 / 지방 영 제20조제1항제12호나목
# 해당 금액대에서는 중소기업자로 제한할 수 있고, 제한하지 않으면 위반.
V16_중소제한 = [
    Signal("중소기업", _p(r"중소기업")),
    Signal("소상공인", _p(r"소상공인")),
    Signal("소기업", _p(r"소기업")),
]

# --------------------------------------------------------------------------- v18 소기업·소상공인 제한 (1억 미만)
# 국가 영 제21조제1항제10호가목 / 지방 영 제20조제1항제12호가목
V18_소기업제한 = [
    Signal("소기업", _p(r"소\s*기업")),
    Signal("소상공인", _p(r"소상공인")),
    Signal("벤처기업", _p(r"벤처기업")),
    Signal("창업기업", _p(r"창업기업")),
]

# --------------------------------------------------------------------------- v20 SW 대기업 참여제한 명시
# 「중소 소프트웨어사업자의 사업 참여 지원에 관한 지침」제3조제2항
#   "입찰공고문 또는 제안요청서에 대기업 참여제한 하한제도 적용 여부(적용 근거 포함)를
#    명시하여야 한다."
V20_대기업제한명시 = [
    Signal("대기업 참여제한", _p(r"대기업[^\n]{0,12}참여\s*(제한|하한)",
                            r"참여\s*제한[^\n]{0,12}대기업",
                            r"대기업인?\s*소프트웨어\s*사업자")),
    Signal("사업금액 하한", _p(r"사업금액의?\s*하한", r"하한\s*제도")),
    Signal("SW진흥법 48조", _p(r"소프트웨어\s*진흥법[^\n]{0,20}제\s*48\s*조",
                            r"제\s*48\s*조[^\n]{0,20}(대기업|참여)")),
    Signal("중소SW사업자 지침", _p(r"중소\s*소프트웨어\s*사업자")),
]

# SW사업인지 판별하는 신호 (v20 적용 대상 판정용)
SW사업_신호 = [
    Signal("소프트웨어", _p(r"소프트웨어", r"(?<![A-Za-z])SW(?![A-Za-z])")),
    Signal("정보화", _p(r"정보화\s*사업", r"정보시스템")),
    Signal("시스템구축", _p(r"시스템\s*(구축|개발|유지관리|운영)")),
    Signal("홈페이지", _p(r"홈페이지\s*(구축|개편|유지)", r"웹사이트\s*구축")),
    Signal("앱개발", _p(r"(어플리케이션|애플리케이션|모바일\s*앱)\s*(개발|구축)")),
]

# 중기간 경쟁제품 입찰인지 판별하는 신호 (v10·v11·v13 적용 대상 판정용)
#
# ⚠️ 여기에 "직접생산확인"을 넣으면 안 된다.
#    v10 은 "직생 요구가 없는 것"이 위반인데, 직생 문구를 경쟁제품 판별 근거로 쓰면
#    (경쟁제품이다 ∧ 직생문구 없다) 가 서로 모순이 되어 v10 이 영원히 0이 된다.
#    dev 측정에서 실제로 v10 F1 이 0.00 으로 떨어졌다. 순환 논리 금지.
#
#    경쟁제품 여부의 1차 근거는 **조달 대상 품목**이며, 중기부고시 세부품명
#    616개와의 대조가 정답에 가깝다 → pps/pumnum.py 가 후보를 만들고
#    최종 판단은 LLM 이 한다. 아래 신호는 보조용이다.
경쟁제품_신호 = [
    Signal("중기간경쟁제품", _p(r"중소기업자\s*간\s*경쟁\s*제품", r"중기간\s*경쟁")),
    Signal("판로지원법9조", _p(r"판로지원법[^\n]{0,20}제\s*9\s*조",
                          r"「?중소기업제품\s*구매촉진[^」\n]{0,30}」?\s*제\s*9\s*조")),
]


REGISTRY: Dict[str, List[Signal]] = {
    "v10": V10_직생요구,
    "v11": V11_중소제한,
    "v16": V16_중소제한,
    "v18": V18_소기업제한,
    "v20": V20_대기업제한명시,
}


# --------------------------------------------------------------------------- 제한 vs 언급
# dev 검증에서 드러난 함정: 공고문은 「중소기업제품 구매촉진 및 판로지원에 관한 법률」
# 같은 상투 문구로 '중소기업'을 늘 언급한다. 언급은 참가자격 제한이 아니다.
# 기업규모 항목(v11·v13·v14·v15·v16·v17·v18)은 '제한했는가'를 봐야 하므로
# 규모 용어 근처에 제한 표현이 함께 오는지로 판정한다.

_제한표현 = _p(
    r"제한", r"한정", r"한함", r"한한다", r"에\s*한하여", r"만\s*참여", r"만\s*입찰",
    r"이어야\s*합?니?다", r"이어야\s*함", r"인\s*업체", r"인\s*자", r"에\s*해당하는\s*자",
    r"소지한?\s*(자|업체)", r"보유한?\s*(자|업체)", r"자격을?\s*갖춘", r"참가자격",
    r"확인서를?\s*(소지|제출|보유)",
)

# 규모 용어가 '법령명'의 일부로 등장하는 경우는 언급으로 본다 (제한 아님).
_법령문맥 = _p(
    r"「[^」\n]{0,40}(중소기업|소상공인)[^」\n]{0,40}」",
    r"판로지원에\s*관한\s*법률",
    r"구매촉진\s*및\s*판로지원",
)


def near(text: str, term: re.Pattern, cue: re.Pattern, window: int = 70) -> Optional[str]:
    """term 매치 주변 window 문자 안에 cue 가 있으면 그 구간을 돌려준다."""
    for m in term.finditer(text):
        s = max(0, m.start() - window)
        e = min(len(text), m.end() + window)
        ctx = text[s:e]
        if cue.search(ctx):
            return ctx
    return None


def restricted_to(text: str, term: re.Pattern, window: int = 70) -> Presence:
    """참가자격이 해당 규모로 '제한'되었는지. 단순 언급은 제외한다."""
    for m in term.finditer(text):
        s = max(0, m.start() - window)
        e = min(len(text), m.end() + window)
        ctx = text[s:e]
        if not _제한표현.search(ctx):
            continue
        # 법령명 인용 안에서만 걸린 경우는 언급으로 본다
        around = text[max(0, m.start() - 30): m.end() + 30]
        if _법령문맥.search(around) and not _제한표현.search(
                text[m.end(): m.end() + window]):
            continue
        return Presence("", True, ["제한표현근접"], ctx.strip())
    return Presence("", False, [], None)


_중소기업 = _p(r"중소기업")
_소기업 = _p(r"소\s?기업", r"소상공인", r"벤처기업", r"창업기업")


def size_restrictions(text: str) -> Dict[str, Presence]:
    """기업규모 제한 실태. 판정이 아니라 사실 추출이다."""
    return {
        "중소기업": restricted_to(text, _중소기업),
        "소기업등": restricted_to(text, _소기업),
    }


@dataclass
class Presence:
    item: str
    present: bool
    hits: List[str]          # 걸린 신호 이름
    sample: Optional[str]    # 원문에서 실제로 매칭된 문자열 (디버깅용)


def scan(text: str, signals: Sequence[Signal]) -> Presence:
    hits: List[str] = []
    sample: Optional[str] = None
    for s in signals:
        m = s.pattern.search(text)
        if m:
            hits.append(s.name)
            if sample is None:
                sample = m.group(0)
    return Presence("", bool(hits), hits, sample)


def scan_record(rec: Record) -> Dict[str, Presence]:
    """레코드 전체 원문(절단 없음)에 대해 부재탐지 신호를 스캔한다."""
    text = rec.full_text
    out: Dict[str, Presence] = {}
    for item, signals in REGISTRY.items():
        p = scan(text, signals)
        out[item] = Presence(item, p.present, p.hits, p.sample)
    out["_SW사업"] = scan(text, SW사업_신호)
    out["_경쟁제품"] = scan(text, 경쟁제품_신호)
    return out


def absence_violation_candidates(rec: Record) -> Dict[str, bool]:
    """정규식만으로 본 부재 여부. True = '요구 문구가 없다' = 위반 후보.

    적용 대상 여부(경쟁제품 입찰인가·SW사업인가·금액대가 맞는가)는
    gating.py 와 LLM 이 따로 판단한다. 여기서는 존재/부재만 본다.
    """
    p = scan_record(rec)
    return {item: (not p[item].present) for item in REGISTRY}
