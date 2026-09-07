# -*- coding: utf-8 -*-
"""항목 그룹별 프롬프트·스키마.

24항목 한 방 프롬프트를 버리고 4개 그룹으로 나눈 이유
  고정 모델(gemma-4-26B-A4B-it)은 추론력은 31B와 대등하지만(AIME 88.3 vs 89.2,
  GPQA 82.3 vs 84.3) **긴 문맥 탐색과 다단계 합성이 약하다**
  (MRCR 44.1 vs 66.4, BBEH 64.8 vs 74.4, Tau2 68.2 vs 76.9).
  → 판단 난이도는 유지하고, 문맥 길이와 동시 처리 항목 수를 줄인다.
    "어려운 판단 하나를 짧은 텍스트로 묻는다"가 이 모델의 최적점이다.

게이팅에서 제외된 항목은 프롬프트와 스키마 양쪽에서 아예 뺀다.
물어보지 않은 항목은 틀릴 수 없고, 토큰도 아낀다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from . import compare, gosimatch, law, presence, pumnum, schedule, sections, spec
from .records import ABSENCE, Record


@dataclass(frozen=True)
class Group:
    key: str
    items: Sequence[str]
    cue: str                    # sections.CUES 키
    budget: int                 # 문서 텍스트 글자 예산
    지시: str                    # 그룹 고유 판정 지침
    full_doc: bool = False      # 부재탐지 포함 → 전체 문서 커버리지 필요


GROUPS: List[Group] = [
    Group(
        key="G1자격",
        items=("v1", "v2", "v3", "v4", "v5", "v6", "v7", "v8"),
        cue="자격",
        budget=4000,
        지시="""이 묶음은 **입찰참가자격 제한**이 법령이 허용하는 범위를 넘었는지 본다.

판정 기준
- v1 특정기관 제한: 대학·산학협력단·특정 협회·특정 인증기관 등 **특정 기관에 소속된 자만**
  참여할 수 있게 했으면 위반. 업종·면허 등록 요구는 여기에 해당하지 않는다.
- v2 고시금액 미만 실적제한: 추정가격이 고시금액 미만인 제조·용역인데 **이행실적을 요구**하면 위반.
- v3 실적제한 배수 초과: 요구 실적금액이 이 사업 추정가격의 허용 배수를 넘으면 위반.
- v4 특정기관·특정실적: "국가기관이 발주한", "대학병원에 납품한" 처럼 **실적의 발주처·납품처를
  특정**하면 위반.
- v5 기준금액 이상 지역제한: 지역제한이 허용되지 않는 금액대인데 소재지를 제한하면 위반.
- v6 시·군·구 지역제한: 지역제한을 시·도가 아니라 **시·군·구 단위로 좁히면** 위반.
- v7 인접 시·도 확대: 지역제한을 **둘 이상의 시·도로 넓히면** 위반(법정 예외 사유가 공고에
  명시돼 있으면 위반 아님).
- v8 중복제한: **실적제한과 지역제한이 둘 다 참가자격으로 요구**될 때만 위반.
  지역제한만 있거나 실적제한만 있으면 v8 이 아니다(각각 v5~v7 또는 v2~v3 에서 본다).
  근거문구는 둘 중 하나가 아니라 두 제한이 함께 보이는 부분을 인용한다.""",
    ),
    Group(
        key="G2기업규모",
        items=("v10", "v11", "v12", "v13", "v14", "v15", "v16", "v17", "v18"),
        cue="기업규모",
        budget=4000,
        full_doc=True,
        지시="""이 묶음은 **기업규모 제한과 직접생산 요구**가 금액대·품목에 맞는지 본다.

먼저 이 입찰이 «중기간 경쟁제품» 입찰인지 정한다.
**공고문이 뭐라고 쓰여 있는지가 아니라, 무엇을 조달하는지로 판단하라.**
아래 [중기부고시 경쟁제품 대조]의 조달 대상과 고시 목록을 견주어 결정한다.
공고문이 "중소기업자간 경쟁제품"이라고 밝히지 않았어도 품목이 해당하면 경쟁제품 입찰이며,
밝히지 않은 채 요건을 걸지 않은 것이 오히려 전형적인 위반이다.

경쟁제품 입찰인 경우
- v10 직접생산확인증명서 소지를 **요구하지 않았으면** 위반.
- v11 참가자격을 **중소기업자로 제한하지 않았으면** 위반.
- v13 참가자격을 **소기업·소상공인만으로 좁혔으면** 위반.
  "중소기업 또는 소상공인", "중기업·소기업·소상공인" 처럼 **중기업이 포함되면 v13 이 아니다.**

경쟁제품이 아닌 일반 물품·용역인 경우 — 추정가격 구간이 허용 범위를 정한다.
- v12 경쟁제품이 아닌데 **직접생산확인을 요구**하면 위반.
- v14 고시금액 이상인데 중소기업으로 제한하면 위반.
- v15 1억~고시금액 구간인데 **소기업·소상공인만으로 좁히면** 위반(중기업 포함이면 아님).
- v16 1억~고시금액 구간인데 **중소기업 제한을 아예 안 했으면** 위반.
- v17 1억 미만인데 **중소기업으로 제한**하면 위반(이 구간은 소기업·소상공인이 대상).
- v18 1억 미만인데 **소기업·소상공인 제한을 아예 안 했으면** 위반.

⚠️ 「중소기업제품 구매촉진 및 판로지원에 관한 법률」 같은 **법령 이름을 인용한 것만으로는
   제한한 것이 아니다.** 참가자격으로 실제로 요구했는지를 보라.""",
    ),
    Group(
        key="G3물품SW공동",
        items=("v9", "v19", "v20", "v21"),
        cue="물품SW공동",
        budget=4000,
        full_doc=True,
        지시="""판정 기준
- v9 특정 모델명: 규격서·과업지시서에 **특정 제조사·브랜드·모델명을 지정**하면 위반.
  "동등 이상", "동급 제품 가능" 같은 대체 허용 문구가 함께 있으면 위반이 아니다.
- v19 물품공급 확약서: 제조사·공급사의 **물품공급 확약서·협약서를 입찰 단계에서 제출**하도록
  요구하면 위반(계약 체결 시 제출은 허용).
- v20 SW 대기업 참여제한 명시: 소프트웨어사업인데 공고문·제안요청서에 **대기업 참여제한
  하한제도 적용 여부를 명시하지 않았으면** 위반. 소프트웨어사업이 아니면 위반이 아니다.
- v21 공동수급 최소지분율: 공동수급체 구성원의 **최소지분율을 법정 하한보다 낮게** 정하면 위반.""",
    ),
    Group(
        key="G4설명회대조",
        items=("v22", "v23", "v24"),
        cue="설명회대조",
        budget=3500,
        지시="""판정 기준
- v22 현장설명회 참석 제한: **설명회에 참석한 업체만 입찰·제안서 제출이 가능**하도록 하면 위반.
  근거 조문이 삭제되어 참석을 자격요건으로 삼을 수 없다. 단순히 설명회를 여는 것은 위반이 아니다.
- v23 현장설명회 공고기간: 설명회를 여는 경우 **설명회일부터 제안서 제출마감일까지**
  법정 기간(추정가격 1억 미만 10일 / 1억~10억 20일 / 10억 이상 40일)이 확보되어야 하고,
  입찰공고는 설명회일 7일 전까지 이루어져야 한다. 둘 중 하나라도 모자라면 위반.
  ※ 입찰공고 자체가 긴 것은 문제가 아니다. **설명회를 늦게 잡아 준비기간이 짧아진 것**이 위반이다.
  아래 [일정 검산] 결과를 우선하고, '판단 보류'인 경우에만 문서를 직접 읽어 판단하라.
- v24 공고서와 나라장터 등록값 상이: [나라장터 등록정보]의 값과 공고문 본문의 기재가
  **실제로 다른 값**일 때만 위반. 대조 대상은 예산·추정가격, 계약방법, 낙찰방법, 지역제한, 업종제한.
  · 금액은 "금액 대조(계산됨)" 결과를 따른다. **일치로 계산된 금액을 인용해 위반이라 하지 않는다.**
  · 금액이 적혀 있다는 사실만으로는 위반이 아니다. 등록값과 **다른 숫자**여야 한다.
  · 본문에 언급이 없는 항목은 불일치로 보지 않는다.
  · 근거문구는 등록값과 다른 값이 적힌 바로 그 부분을 인용한다.""",
    ),
]

GROUP_OF: Dict[str, Group] = {i: g for g in GROUPS for i in g.items}


# --------------------------------------------------------------------------- 스키마

def build_schema(items: Sequence[str]) -> Dict[str, Any]:
    """제약 디코딩용 JSON Schema. 요청한 항목만 포함한다.

    부재탐지 항목은 근거문구를 null 로 고정한다 — 인용할 원문이 없다.
    """
    props: Dict[str, Any] = {}
    for v in items:
        props[v] = {
            "type": "object",
            "additionalProperties": False,
            "required": ["위반여부", "근거문구"],
            "properties": {
                "위반여부": {"type": "integer", "enum": [0, 1]},
                "근거문구": ({"type": "null"} if v in ABSENCE
                          else {"type": ["string", "null"], "maxLength": 600}),
            },
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": list(items),
        "properties": props,
    }


# --------------------------------------------------------------------------- 사실 요약

_MONEY = "{:,}원"


def fact_block(rec: Record) -> str:
    """판정에 직접 쓰이는 사실만 앞에 모아 준다.

    메타 21개 필드를 밋밋하게 나열하면 모델이 중요한 값을 놓친다.
    금액 구간처럼 **계산해서 알려줄 수 있는 것은 계산해서** 준다 —
    이 모델은 추론은 하지만 긴 목록에서 값을 찾아내는 데 약하다.
    """
    상한 = law.고시금액(rec.업무구분)
    밴드설명 = {
        "A": f"추정가격이 1억원 미만 → 소기업·소상공인·벤처·창업기업까지만 제한 가능",
        "B": f"추정가격이 1억원 이상 고시금액({상한:,}원) 미만 → 중소기업자까지만 제한 가능",
        "C": f"추정가격이 고시금액({상한:,}원) 이상 → 기업규모 제한 불가",
    }[rec.판로지원밴드]

    lines = [
        f"- 적용 계약법: {rec.적용계약법}",
        f"- 업무구분: {rec.업무구분} · 계약방법: {rec.계약방법} · 낙찰방법: {rec.낙찰방법}",
        f"- 입찰추정가격: {_MONEY.format(rec.추정가격)}"
        + (f" · 배정예산금액: {_MONEY.format(rec.배정예산)}" if rec.배정예산 else ""),
        f"- 기업규모 제한 허용범위: {밴드설명}",
        f"- 실적제한 허용 배수: 해당 사업 추정가격의 "
        + ("1배 이내(금액기준)" if not rec.is_지방 else "1배 이내(금액기준), 규모·양은 1/3 이내"),
    ]
    # 지역제한 허용 여부 — 계산해서 준다. dev 측정에서 모델이 7.27억 용역의
    # 서울 소재 제한을 위반으로 보지 못했다(v5 놓침).
    if rec.is_공사:
        지역기준 = law.지역제한_기준_지방_공사_종합 if rec.is_지방 else law.고시금액_공사
    elif rec.is_지방:
        지역기준 = law.지역제한_기준_지방_물품용역
    else:
        지역기준 = law.지역제한_기준_국가_물품용역
    if rec.추정가격 >= 지역기준:
        lines.append(f"- 지역제한 허용 여부: 추정가격이 기준금액({지역기준:,}원) 이상 → "
                     f"**지역제한 불가**. 소재지 제한이 있으면 v5 위반")
    else:
        lines.append(f"- 지역제한 허용 여부: 기준금액({지역기준:,}원) 미만 → 시·도 단위 제한 가능. "
                     f"시·군·구로 좁히면 v6, 둘 이상 시·도로 넓히면 v7")

    if rec.meta.get("조항호내용"):
        lines.append(f"- 발주기관이 등록한 제한 조항호: {rec.meta['조항호내용']}")
        why = compare.interpret_clause(rec)
        if why:
            lines.append(f"  → 해석: {why}")
    for k in ("지역제한여부", "업종제한여부", "면허업종제한목록",
              "공동도급구성방식", "세부품명번호목록", "긴급공고여부",
              "공고게시일자", "개찰예정일자"):
        v = rec.meta.get(k)
        if v not in (None, ""):
            lines.append(f"- {k}: {v}")
    lines.append("- " + compare.region_check(rec))
    amt_txt, _ = compare.amount_check(rec)
    lines.append(f"- 금액 대조(계산됨): {amt_txt}")
    return "\n".join(lines)


def item_block(items: Sequence[str], tbl: Dict[str, Dict[str, Any]]) -> str:
    out = []
    for v in items:
        it = tbl.get(v, {})
        name = it.get("항목명", v)
        tag = "  [부재탐지 — 근거문구는 null]" if v in ABSENCE else ""
        note = f" (참고: {it['비고']})" if it.get("비고") else ""
        out.append(f"- {v}: {name}{note}{tag}")
    return "\n".join(out)


# --------------------------------------------------------------------------- 프롬프트

SYSTEM = """당신은 공공 입찰공고의 법령 위반 여부를 점검하는 계약 심사관이다.

지켜야 할 것
1. 요청받은 항목 전부에 답한다. 애매하면 0(위반 아님)으로 낸다.
2. 근거 문구는 **주어진 문서에 그대로 있는 문장**을 옮긴다. 요약·수정·생성 금지.
   위반에 해당하는 부분만 짧게 인용한다.
3. [부재탐지] 표시 항목은 **있어야 할 기재가 없는 것**이 위반이다.
   인용할 원문이 없으므로 근거문구는 null 로 둔다.
4. 문서에 실제로 적힌 것만 근거로 삼는다. 관행이나 추측으로 판단하지 않는다.
5. [나라장터 등록정보]에 **계산된 사실**(금액 구간·지역제한 허용 여부·금액 대조 결과)이
   있으면 그것을 우선한다. 스스로 다시 계산하지 않는다.

익명화 토큰 읽는 법 (기관명·지역명이 치환되어 있다)
- [지역:r1|단위=기초|광역=경기도] = 경기도 안의 **시·군·구 하나**. '단위=기초'면 시·군·구 단위다.
- [지역:r1|단위=광역|…] 또는 광역명이 그대로 적힌 것 = 시·도 단위.
- 같은 문서에서 r1, r2 는 서로 다른 지역이다.
- [수요기관(기초자치단체)|지역=r1] = 발주기관 유형. 자격 제한이 아니다.

출력은 JSON 하나만. 설명·머리말을 덧붙이지 않는다."""


# --------------------------------------------------------------------------- 2단계 검증

VERIFY_SYSTEM = """당신은 공공 입찰공고 법령 위반 판정을 **재검토**하는 심사관이다.

앞선 심사에서 어떤 항목이 위반(1)으로 판정되었다. 그 판정이 옳은지 다시 본다.
재검토의 목적은 **과잉 판정을 걸러내는 것**이다. 실제 위반은 드물게 발견되는 것이므로,
근거가 확실하지 않으면 위반이 아니라고 판단해야 한다.

다음을 확인한다.
1. 제시된 근거 문구가 **그 항목이 규정한 위반**을 실제로 보여주는가?
   다른 항목의 소관이거나, 단순히 관련 용어가 등장한 것뿐이면 위반이 아니다.
2. 이 공고의 금액대·계약법·조달 대상에서 그 행위가 **실제로 금지**되는가?
   법이 허용하는 범위 안이면 위반이 아니다.
3. 법정 예외에 해당하지 않는가?

출력은 JSON 하나만:
{"위반유지": 0 또는 1, "이유": "한 문장"}
확실한 위반이면 1, 아니면 0."""


def build_verify_messages(
    rec: Record,
    item: str,
    evidence: Optional[str],
    tbl: Dict[str, Dict[str, Any]],
) -> List[Dict[str, str]]:
    """한 항목의 위반 판정을 재검토하는 대화."""
    it = tbl.get(item, {})
    group = GROUP_OF[item]
    absence = item in ABSENCE

    lines = [
        f"[재검토 항목] {item}: {it.get('항목명', item)}",
    ]
    if it.get("비고"):
        lines.append(f"[항목 참고사항] {it['비고']}")
    lines.append(f"\n[이 항목의 판정 기준]\n{group.지시}")
    lines.append(f"\n[나라장터 등록정보]\n{fact_block(rec)}")

    if absence:
        lines.append(
            f"\n[앞선 판정] 위반 = 1"
            f"\n이 항목은 **부재탐지**다. '있어야 할 기재가 없는 것'이 위반이므로 근거 문구가 없다."
            f"\n확인할 것: ① 이 공고가 정말 그 기재를 해야 하는 대상인가"
            f" ② 문서 어딘가에 그 기재가 실제로 있지 않은가"
            f" ③ 법정 예외에 해당하지 않는가")
        doc = sections.full_context(rec, 5000)
    else:
        lines.append(f"\n[앞선 판정] 위반 = 1"
                     f"\n[제시된 근거 문구]\n{evidence or '(근거 문구가 제시되지 않았다)'}")
        if not evidence:
            lines.append("근거 문구가 없다면 위반을 입증하지 못한 것이다.")
        doc = sections.select(rec, sections.CUES[group.cue], 4000)

    lines.append(f"\n[공고 문서]\n{doc}")
    lines.append("\n이 판정을 유지해야 하는가? JSON 으로 답하라.")

    return [
        {"role": "system", "content": VERIFY_SYSTEM},
        {"role": "user", "content": "\n".join(lines)},
    ]


VERIFY_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["위반유지", "이유"],
    "properties": {
        "위반유지": {"type": "integer", "enum": [0, 1]},
        "이유": {"type": "string", "maxLength": 300},
    },
}


def build_messages(
    rec: Record,
    group: Group,
    items: Sequence[str],
    tbl: Dict[str, Dict[str, Any]],
    gosi: Optional[pumnum.Gosi] = None,
    law_text: str = "",
    budget: Optional[int] = None,
) -> List[Dict[str, str]]:
    """한 그룹에 대한 대화 메시지."""
    b = budget if budget is not None else group.budget

    if group.full_doc:
        doc = sections.full_context(rec, b)
    else:
        doc = sections.select(rec, sections.CUES[group.cue], b)

    parts = [
        f"[검토 항목]\n{item_block(items, tbl)}",
        f"\n[판정 지침]\n{group.지시}",
    ]
    if law_text:
        parts.append(f"\n[관련 법령 조문]\n{law_text}")
    parts.append(f"\n[나라장터 등록정보]\n{fact_block(rec)}")

    if group.key == "G2기업규모" and gosi is not None:
        parts.append(f"\n[중기부고시 경쟁제품 대조]\n{gosimatch.block(rec, gosi)}")
    if group.key == "G4설명회대조" and "v23" in items:
        parts.append(f"\n[일정 검산]\n{schedule.fact_line(rec)}")
    if group.key == "G3물품SW공동" and "v9" in items:
        # 모델명은 첨부 깊숙이 있어 섹션 파서가 놓친다(dev 입력누락 2건) → 후보를 따로 뽑아 준다
        parts.append(f"\n[첨부에서 찾은 모델명·제조사 후보 줄 (v9)]\n{spec.candidate_block(rec)}")
    if group.key == "G3물품SW공동" and "v21" in items:
        # 모델은 v21 기준("법정 하한보다 낮으면 위반")은 받지만 하한값을 모른다.
        # 지방 5% / 국가 10% 이고 20% 가감이 허용되므로 실효 하한은 4% / 8% —
        # 조문에서 계산되는 값이라 코드가 준다. 판정은 강제하지 않는다.
        parts.append(f"\n[공동수급 최소지분율 (v21, 계산됨)]\n{compare.joint_share_check(rec)}")

    parts.append(f"\n[공고 문서]\n{doc}")
    parts.append(
        "\n위 항목 각각에 대해 위반여부(0 또는 1)와 근거문구를 JSON으로 내라.")

    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": "\n".join(parts)},
    ]
