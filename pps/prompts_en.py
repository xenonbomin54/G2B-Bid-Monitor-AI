# -*- coding: utf-8 -*-
"""영어 지시문판 프롬프트 — A/B 실험 전용 (§prompts 의 한국어판과 1:1 대응).

## 왜 만들었나
평가용 고정 모델(Gemma 계열)이 한국어 지시문보다 영어 지시문을 더 정확히 따를 수 있다는
가설을 시험한다. 지금까지의 실패 이력은 **지시 내용**을 바꾼 실험이었고(프롬프트 5/5 실패),
이 실험은 **내용을 그대로 두고 지시 언어만** 바꾼다. 한 번에 한 변수라는 원칙에 맞다.

## 무엇을 바꾸고 무엇을 그대로 두는가
바꾸는 것 — 지시문뿐이다.
  · SYSTEM · GRADE_GUIDE · 그룹 전문(preamble) · 항목별 판정 기준 · 꼬리 지시문 · dual 지시문
  · 섹션 라벨([검토 항목] 등). SYSTEM 이 라벨을 이름으로 참조하므로 함께 영어로 맞춘다.
그대로 두는 것 — 이것들이 바뀌면 A/B 가 오염된다.
  · 공고 원문 (한국어)
  · 사실 블록의 **내용** (fact_block · gosimatch · schedule · spec · compare 출력, 한국어)
  · 항목명 (항목 정의표에서 오는 한국어)
  · **스키마 키** — "위반등급" · "근거문구" · "적법근거" · "위반항목". 키를 영어로 바꾸면
    스키마가 달라져 파싱 경로와 등급 저장 포맷까지 흔들린다. 동일 스키마를 유지한다.
  · 등급 체계 0~3 · 문턱 · gating · 규칙 · API 설정
  · 법령 명칭 — 「중소기업기본법」 같은 고유명사는 한국어 원문을 유지하고 필요하면 영어를
    괄호로 덧붙인다. 번역하면 공고 원문의 같은 문구와 이어지지 않는다.

## 원문과의 대응
한국어판에 있는 강조(**…**)·경고(⚠️)·순서·줄바꿈 구조를 그대로 옮겼다. 문장을 합치거나
쪼개지 않았다. dev200k 에서 "구조와 문구를 같이 바꾸면 원인을 분리할 수 없다"는 교훈을
얻었으므로, 여기서는 **언어만** 바꾼다.
"""
from __future__ import annotations

from typing import Dict, Sequence

# --------------------------------------------------------------------------- 시스템

SYSTEM = """You are a contract review officer who checks public procurement notices \
for violations of Korean procurement law.

Rules you must follow
1. Answer for every item you are asked about. When in doubt, answer 0 (not a violation).
2. For the evidence quote, copy **a sentence that appears verbatim in the given documents**.
   Do not summarize, edit, or invent. Quote only the part that constitutes the violation.
   The documents are in Korean, so **the quote must stay in Korean, exactly as written**.
3. Items marked [ABSENCE] are violated when **a required statement is missing**.
   There is nothing to quote, so leave the evidence field null.
4. Base your judgment only on what the documents actually say. Do not judge from custom
   or conjecture.
5. When [KONEPS registered data] contains a **computed fact** (price band, whether a regional
   restriction is permitted, the result of an amount comparison), that fact takes precedence.
   Do not recompute it yourself.

How to read the anonymization tokens (agency and place names have been substituted)
- [지역:r1|단위=기초|광역=경기도] = **one city/county/district** inside 경기도.
  '단위=기초' means the municipal (city/county/district) level.
- [지역:r1|단위=광역|…], or a province name written out = the province (시·도) level.
- Within one document, r1 and r2 are different places.
- [수요기관(기초자치단체)|지역=r1] = the type of the ordering agency. It is not a
  qualification restriction.

Output exactly one JSON object. Do not add any explanation or preamble.

Note: the JSON keys are Korean and must be used exactly as given in the schema
("위반등급", "근거문구", "적법근거")."""


GRADE_GUIDE = """
Report your judgment as a **violation grade from 0 to 3**, not as 0/1. The grades mean:
- 3 = clear violation. The evidence quote shows the violation directly.
- 2 = appears to be a violation. You can point to an evidence quote.
- 1 = suspicion. Related wording exists but you cannot conclude it is a violation.
- 0 = not a violation. There is no related wording, or it falls within what the law permits.
For grade 1 or above, give the evidence quote as well (null for [ABSENCE] items).
For grade 0 the evidence quote is null.
Do not set a threshold of your own when grading — just express your degree of \
confidence as the grade."""


# --------------------------------------------------------------------------- 섹션 라벨
# SYSTEM 이 "[KONEPS registered data]" 를 이름으로 참조하므로 여기서 어긋나면 안 된다.

LABELS: Dict[str, str] = {
    "검토 항목": "Items under review",
    "판정 지침": "Assessment criteria",
    "관련 법령 조문": "Relevant statutes",
    "나라장터 등록정보": "KONEPS registered data",
    "중기부고시 경쟁제품 대조": "Cross-check against the MSS designated-product list",
    "일정 검산": "Schedule arithmetic",
    "첨부에서 찾은 모델명·제조사 후보 줄 (v9)": (
        "Candidate lines with model names / manufacturers found in the attachments (v9)"),
    "공동수급 최소지분율 (v21, 계산됨)": (
        "Minimum share for joint-venture members (v21, computed)"),
    "공고 문서": "Notice documents",
}

ABSENCE_TAG = "  [ABSENCE — evidence quote must be null]"
NOTE_PREFIX = "note: "


# --------------------------------------------------------------------------- 그룹 전문

GROUP_PREAMBLE: Dict[str, str] = {
    "G1자격": (
        "This group asks whether the **restrictions on bidder qualification** go beyond "
        "what the law permits."),
    "G2기업규모": """This group asks whether the **firm-size restrictions and \
direct-production requirements** fit the price band and the procured item.

First decide whether this tender is a «중기간 경쟁제품» tender (a product reserved for
competition among small and medium enterprises).
**Decide it from what is being procured, not from what the notice says about itself.**
Compare the procurement target with the designated list in
[Cross-check against the MSS designated-product list] below.
Even if the notice never states "중소기업자간 경쟁제품", it is a reserved-product tender when
the item is on the list — and failing to state it while imposing none of the requirements is
in fact the typical violation.

If it is an ordinary (non-reserved) good or service, **the estimated-price band determines
what is permitted.** Which band this notice falls in has already been computed for you under
'기업규모 제한 허용범위' in [KONEPS registered data].

⚠️ **Merely citing the name of a statute** such as 「중소기업제품 구매촉진 및 판로지원에 관한 \
법률」 **is not itself a restriction.** Look at whether it was actually required as a
qualification to participate.""",
    "G3물품SW공동": """Assessment criteria
- v9 specific model name: it is a violation to **designate a specific manufacturer, brand, or
  model name** in the specification or the statement of work.
  It is not a violation when wording that allows substitutes accompanies it, such as
  "동등 이상" (equivalent or better) or "동급 제품 가능" (an equivalent-class product is acceptable).
- v19 supply undertaking: it is a violation to require a manufacturer's or supplier's
  **supply undertaking or agreement to be submitted at the bidding stage**
  (requiring it at contract signing is permitted).
- v20 SW large-enterprise participation limit: for a software project, it is a violation if the
  notice or the request for proposal **does not state whether the large-enterprise
  participation-limit system applies**. If it is not a software project, it is not a violation.
- v21 minimum joint-venture share: it is a violation to set the **minimum share of a
  joint-venture member below the statutory floor**.""",
    "G4설명회대조": (
        "This group asks about **site-briefing restrictions and scheduling**, and about "
        "**discrepancies between the notice text and the registered data**."),
}


# --------------------------------------------------------------------------- 항목별 기준

ITEM_RULES: Dict[str, str] = {
    "v1": ("- v1 restriction to specific institutions: it is a violation if only those "
           "**affiliated with specific institutions**\n"
           "  — universities, industry-academic cooperation foundations, a particular "
           "association or certification body — may participate.\n"
           "  Requiring a business-type or licence registration does not fall under this item."),
    "v2": ("- v2 performance-record restriction below the notice threshold: it is a violation "
           "to **require a past performance record** when the estimated price of a "
           "manufacturing or service contract is below the notice threshold (고시금액)."),
    "v3": ("- v3 performance-record multiple exceeded: it is a violation if the required "
           "record amount exceeds the permitted multiple of this project's estimated price."),
    "v4": ("- v4 specific orderer / specific record: it is a violation to **specify who "
           "ordered or received** the past work,\n  as in \"국가기관이 발주한\" (ordered by a "
           "state agency) or \"대학병원에 납품한\" (delivered to a university hospital)."),
    "v5": ("- v5 regional restriction at or above the threshold: it is a violation to restrict "
           "the bidder's location when the price band does not permit a regional restriction."),
    "v6": ("- v6 city/county/district-level regional restriction: it is a violation to narrow a "
           "regional restriction **to the municipal (시·군·구) level** instead of the "
           "province (시·도) level."),
    "v7": ("- v7 extension to adjacent provinces: it is a violation to widen a regional "
           "restriction **to two or more provinces**\n  (not a violation if a statutory "
           "exception is stated in the notice)."),
    "v8": ("- v8 overlapping restrictions: a violation only when **both a performance-record "
           "restriction and a regional restriction are required as qualifications**.\n"
           "  If only one of them is present it is not v8 "
           "(that is handled by v5–v7 or v2–v3 respectively).\n"
           "  For the evidence quote, cite the part where **both** restrictions appear "
           "together, not just one of them."),

    "v10": ("- v10 [reserved-product tender] a violation if a direct-production confirmation "
            "certificate (직접생산확인증명서) **was not required**."),
    "v11": ("- v11 [reserved-product tender] a violation if participation **was not restricted "
            "to small and medium enterprises (중소기업자)**."),
    "v13": ("- v13 [reserved-product tender] a violation if participation was **narrowed to "
            "소기업·소상공인 only** (small enterprises and micro-businesses).\n"
            "  When **중기업 (medium enterprises) are included** — as in "
            "\"중소기업 또는 소상공인\" or \"중기업·소기업·소상공인\" — **it is not v13.**"),
    "v12": ("- v12 [ordinary good or service] a violation to **require direct-production "
            "confirmation** when this is not a reserved product."),
    "v14": ("- v14 [ordinary good or service · at or above the notice threshold] "
            "a violation to restrict participation to 중소기업 (SMEs)."),
    "v15": ("- v15 [ordinary good or service · 100 million won to the notice threshold] "
            "a violation to **narrow participation to 소기업·소상공인 only**.\n"
            "  What this band permits reaches **up to 중소기업자 (SMEs)**. Therefore a "
            "restriction that includes 중소기업, such as \"중소기업 또는 소상공인\", "
            "**is lawful and is not v15.**"),
    "v16": ("- v16 [ordinary good or service · 100 million won to the notice threshold] "
            "a violation if **no 중소기업 restriction was imposed at all**.\n"
            "  If a restriction was imposed, it is lawful as long as its scope reaches "
            "up to 중소기업자.\n"
            "  ※ **Statutory exception** — Article 2-3 of the Enforcement Decree of the "
            "판로지원법 provides that\n"
            "    the agency **may choose not to** conclude a priority-procurement contract "
            "in the cases below, and\n"
            "    paragraph 2 of the same Article requires **the reason to be stated in the "
            "notice**. If the notice\n"
            "    gives such a reason, the absence of a 중소기업 restriction "
            "**is not a violation.**\n"
            "    - fewer than two SMEs bid, or the tender failed because no qualified "
            "bidder appeared\n"
            "    - advisory services through intellectual activity such as research, study, "
            "survey, inspection,\n"
            "      evaluation or development; school educational activities outside the "
            "curriculum; specimen testing;\n"
            "      or entrusted execution of a subsidized project — where participation by a "
            "non-profit corporation is needed\n"
            "    - goods or services that another statute designates for priority purchase, "
            "or for which it permits\n"
            "      a private contract or designated competition\n"
            "    - a particular performance, technology or quality is required, so the purpose "
            "cannot be achieved\n"
            "      through the priority-procurement route\n"
            "    ⚠️ **Citing the article number alone is not enough.** You must be able to "
            "identify which of the\n"
            "      above reasons applies from the notice itself. If the reason cannot be "
            "confirmed, do not treat it\n"
            "      as an exception."),
    "v17": ("- v17 [ordinary good or service · below 100 million won] a violation to "
            "**restrict participation to 중소기업**.\n"
            "  What this band permits reaches **up to 소기업·소상공인**. Since 중소기업 "
            "includes 중기업,\n"
            "  opening participation up to 중소기업 goes beyond the permitted scope.\n"
            "  **Listing them together is also v17** — as in \"중소기업 또는 소상공인\" or "
            "\"중·소기업, 소상공인\" —\n"
            "  because the fact that participation was opened up to 중소기업 remains true "
            "even when 소상공인\n"
            "  is written alongside. Do not treat the joint listing as grounds for lawfulness.\n"
            "  **Do not ask which section it appears in.** A restriction does not have to sit "
            "under a heading\n"
            "  called '입찰참가자격'. In practice it is often written indirectly, as "
            "\"중소기업·소상공인 확인 대상\"\n"
            "  (subject to SME/micro-business verification) or "
            "\"중소기업·소상공인확인서 제출\" (submit the SME\n"
            "  verification certificate), and that too ultimately means **only firms of that "
            "size may participate**,\n"
            "  so it is v17.\n"
            "  The test is not location but **function** — does that sentence bind whoever "
            "wants to bid on this\n"
            "  tender? If so, it is v17.\n"
            "  Conversely, if it is a post-award duty applying only to the successful bidder or "
            "the contractor,\n"
            "  a description of a different project, a citation of a statute, or a statistical "
            "or informational\n"
            "  remark — anything that **does not govern participation itself** — it is not v17.\n"
            "  ※ This joint-listing test applies **to v17 only**. For v13 (reserved products) "
            "the same joint\n"
            "    listing means the opposite: it is *not* a violation, because 중기업 are "
            "included. When the band\n"
            "    differs, the conclusion is reversed."),
    "v18": ("- v18 [ordinary good or service · below 100 million won] a violation if "
            "**no 소기업·소상공인 restriction was imposed at all**.\n"
            "  ※ **Statutory exception** — Article 2-3 of the Enforcement Decree of the "
            "판로지원법 provides that\n"
            "    the agency **may choose not to** conclude a priority-procurement contract "
            "in the cases below, and\n"
            "    paragraph 2 of the same Article requires **the reason to be stated in the "
            "notice**. If the notice\n"
            "    gives such a reason, the absence of a 소기업·소상공인 restriction "
            "**is not a violation.**\n"
            "    - fewer than two 소기업·소상공인 bid, or the tender failed because no "
            "qualified bidder appeared,\n"
            "      or it is evident that three or fewer qualified 소기업·소상공인 exist\n"
            "    - advisory services through intellectual activity such as research, study, "
            "survey, inspection,\n"
            "      evaluation or development; school educational activities outside the "
            "curriculum; specimen testing;\n"
            "      or entrusted execution of a subsidized project — where participation by a "
            "non-profit corporation is needed\n"
            "    - goods or services that another statute designates for priority purchase, "
            "or for which it permits\n"
            "      a private contract or designated competition\n"
            "    - a particular performance, technology or quality is required, so the purpose "
            "cannot be achieved\n"
            "      through the priority-procurement route\n"
            "    ⚠️ **Citing the article number alone is not enough.** You must be able to "
            "identify which of the\n"
            "      above reasons applies from the notice itself. If the reason cannot be "
            "confirmed, do not treat it\n"
            "      as an exception."),

    "v22": ("- v22 restriction to site-briefing attendees: it is a violation to allow "
            "**only firms that attended the briefing to bid or submit a proposal**.\n"
            "  The provision that once allowed this has been repealed, so attendance cannot be "
            "made a qualification requirement. Simply holding a briefing is not a violation."),
    "v23": ("- v23 site-briefing and notice period: when a briefing is held, the statutory "
            "period must be secured\n"
            "  **from the briefing date to the proposal submission deadline**\n"
            "  (10 days for an estimated price below 100 million won / 20 days for 100 million "
            "to 1 billion / 40 days at or above 1 billion),\n"
            "  and the tender notice must be issued at least 7 days before the briefing date. "
            "If either falls short, it is a violation.\n"
            "  ※ A long notice period is not itself a problem. The violation is "
            "**scheduling the briefing late so that the preparation period becomes short**.\n"
            "  Give precedence to the [Schedule arithmetic] result below, and read the "
            "documents yourself only when it says '판단 보류' (judgment deferred)."),
    "v24": ("- v24 notice text differs from the KONEPS registered data: a violation only when a "
            "value in [KONEPS registered data]\n"
            "  and the statement in the body of the notice are **actually different values**. "
            "The fields to compare are\n"
            "  budget / estimated price, contract method, award method, regional restriction, "
            "and business-type restriction.\n"
            "  · For amounts, follow the \"금액 대조(계산됨)\" (amount comparison, computed) "
            "result.\n"
            "    **Do not cite an amount that was computed as matching and call it a violation.**\n"
            "  · The mere fact that an amount is written down is not a violation. It must be a "
            "**different number** from the registered value.\n"
            "  · A field that the body does not mention is not treated as a mismatch.\n"
            "  · For the evidence quote, cite exactly the part where the differing value is "
            "written."),
}


# --------------------------------------------------------------------------- 꼬리 지시문

TAIL_GRADED = ("\nFor each item above, output the 위반등급 (violation grade, 0–3) and the "
               "근거문구 (evidence quote) as JSON.")

TAIL_BINARY = ("\nFor each item above, output the 위반여부 (violation, 0 or 1) and the "
               "근거문구 (evidence quote) as JSON.")


def tail_dual(dual_items: Sequence[str], grade_key: str, lawful_key: str) -> str:
    """선택적 dual 꼬리 — 한국어판과 문장 순서·강조를 맞춘다."""
    return (
        f"\nFor each item above, output the {grade_key} (violation grade, 0–3) and the "
        f"근거문구 (evidence quote) as JSON.\n"
        f"But **for {', '.join(dual_items)} only**, write `{lawful_key}` **before** you decide "
        f"the grade — find grounds for believing that this notice **stays within what the law "
        f"permits** for that item and quote them verbatim from the Korean text, or state in "
        f"**80 characters or less** why that item does not apply to this notice. "
        f"Use null only when you find nothing.\n"
        f"Firm-size restrictions have a different permitted scope in each price band — if the "
        f"scope the notice required is the same as the scope permitted in that band, it is not "
        f"a violation. That is what you are checking first."
    )


def tail_dual_all(grade_key: str, lawful_key: str) -> str:
    """전체 dual 꼬리 — 한국어판 §dual 전체 적용판에 대응(현재 미사용, 대응 유지용)."""
    return (
        f"\nAnswer for each item above **in this order**.\n"
        f"1. `{lawful_key}` — first find grounds for believing this notice **complied with the "
        f"law** on that item. Quote the lawfully written text verbatim from the Korean, or "
        f"state why the item does not apply to this notice. Keep it **within 80 characters**.\n"
        f"Use null only when you find nothing.\n"
        f"2. `{grade_key}` (0–3) — decide it after writing 1. If the lawful grounds are clear, "
        f"it is 0 or 1.\n"
        f"3. `근거문구` — when the grade is 2 or above, quote the grounds verbatim.\n"
        f"Public tender notices are drafted with reference to the statutes, so most items are "
        f"lawful."
    )


TAIL_SELECT = (
    "\nOf the items above, pick **only the ones that are violations** and put them in the "
    "`{key}` array.\n"
    "Each element has the form {{\"항목\": \"vNN\", \"근거문구\": \"verbatim Korean quote\"}}.\n"
    "**If not a single item applies, output an empty array `[]`** — that is a normal answer.\n"
    "Public tender notices are drafted with reference to the statutes, so many notices have no "
    "violation at all.\n"
    "Items you do not pick are treated as 'not a violation', so pick only the ones you are "
    "sure about."
)


# --------------------------------------------------------------------------- 조립

def item_block(items: Sequence[str], tbl: Dict[str, Dict[str, object]],
               absence: frozenset) -> str:
    """[Items under review] 목록. **항목명은 한국어 원문을 유지한다** — 항목 정의표의 값이고,
    번역하면 판정 기준과 항목명이 어긋난다."""
    out = []
    for v in items:
        it = tbl.get(v, {})
        name = it.get("항목명", v)
        tag = ABSENCE_TAG if v in absence else ""
        note = f" ({NOTE_PREFIX}{it['비고']})" if it.get("비고") else ""
        out.append(f"- {v}: {name}{note}{tag}")
    return "\n".join(out)


def 지침(group_key: str, group_items: Sequence[str], items: Sequence[str]) -> str:
    """한국어판 `Group.지침()` 과 같은 규칙으로 영어 기준을 조립한다.

    항목별 기준이 없는 그룹(G3)은 전문만 돌려준다 — 한국어판과 동일한 동작이다.
    """
    pre = GROUP_PREAMBLE[group_key]
    want = [i for i in group_items if i in set(items) and i in ITEM_RULES]
    if group_key == "G3물품SW공동" or not want:
        return pre
    body = "\n".join(ITEM_RULES[i] for i in want)
    return (f"{pre}\n\n"
            f"Per-item criteria — **judge only the items listed below.**\n"
            f"Items not listed here do not apply to this notice, so do not carry their "
            f"criteria over to judge the ones that are listed.\n{body}")
