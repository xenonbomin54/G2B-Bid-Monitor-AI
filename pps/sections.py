# -*- coding: utf-8 -*-
"""공고문 섹션 파서 — 프롬프트에 넣을 텍스트를 관련 구간으로 압축한다.

왜 필요한가
  · 대회 고정 모델(gemma-4-26B-A4B-it)은 긴 문맥에서 needle 찾기가 약하다.
    MRCR v2 8-needle 128k = 44.1% (31B 덴스는 66.4%). 수만 자를 통째로 던지면
    조항을 못 찾는다 → 관련 절만 골라 밀도를 올려야 한다.
  · 부재탐지 5항목(v10·v11·v16·v18·v20)은 "있어야 할 기재가 없는 것"이 위반이다.
    잘라서 보면 '진짜 없음'과 '잘려서 안 보임'을 구분할 수 없다.
    → 이 항목들에는 full_context() 로 전체 커버리지를 주고, 잘렸으면 명시한다.

설계상 반드시 지킬 것 (dev 측정에서 실제로 깨졌던 것들)
  1. 절은 **상위 단위로만** 자른다. `가. 나. 라.` 같은 하위 항목까지 쪼개면
     하위 조각이 부모 절의 신호어("입찰참가자격")를 잃고 관련도 0으로 탈락한다.
  2. 선택된 절을 다시 이을 때 **인접한 절 사이에는 아무것도 끼워 넣지 않는다.**
     태그를 삽입하면 원문 연속성이 깨져 근거문구가 부분문자열 검사에서 탈락한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .records import DOC_ORDER, Record

# --------------------------------------------------------------------------- 절 머리 인식

# 상위 절만 잡는다. `가.` `①` `○` 같은 하위 항목은 절 내부 내용으로 남긴다.
_TOP_HEAD = re.compile(
    r"^[ \t]*("
    r"\d{1,2}\s*[.)]\s"                   # "1. " "2) "
    r"|[Ⅰ-Ⅹ]+\s*[.)]"                    # 로마숫자
    r"|제\s?\d+\s?[조장절]"                # 제3조 제1장
    r"|[■□▣◈]\s*"                        # 굵은 불릿 대제목
    r"|【[^】\n]{1,20}】"                  # 【입찰참가자격】
    r"|<[^>\n]{1,20}>\s*$"                # <입찰참가자격>
    r")\s*(\S[^\n]{0,60})?",
    re.M,
)

# 절 제목이 아니라 본문 한 줄인 경우를 걸러낸다 (예: "1. 입찰에 부치는 사항"은 제목,
# "2026. 3. 11. 15:00" 은 날짜) — 숫자로만 이어지는 머리는 제외.
_DATEISH = re.compile(r"^\s*\d{1,2}\s*[.)]\s*\d")


@dataclass
class Section:
    doc_type: str
    doc_id: str
    title: str
    start: int          # 해당 문서 text 내 offset
    end: int
    text: str

    def __len__(self) -> int:
        return len(self.text)


def split_sections(text: str, doc_type: str = "", doc_id: str = "",
                   min_len: int = 120) -> List[Section]:
    """상위 절 기준으로 문서를 자른다. 머리를 못 찾으면 문서 전체가 한 절."""
    heads = []
    for m in _TOP_HEAD.finditer(text):
        line = text[m.start():m.start() + 24]
        if _DATEISH.match(line):
            continue
        title = re.sub(r"\s+", " ", (m.group(2) or "")).strip()
        heads.append((m.start(), title))

    if not heads:
        return [Section(doc_type, doc_id, "", 0, len(text), text)]

    out: List[Section] = []
    if heads[0][0] > 0:
        out.append(Section(doc_type, doc_id, "(머리말)", 0, heads[0][0], text[:heads[0][0]]))
    for i, (pos, title) in enumerate(heads):
        end = heads[i + 1][0] if i + 1 < len(heads) else len(text)
        body = text[pos:end]
        # 너무 짧은 절은 앞 절에 합친다 — 표 조각이 잘게 쪼개지는 걸 막는다
        if len(body) < min_len and out:
            prev = out[-1]
            out[-1] = Section(prev.doc_type, prev.doc_id, prev.title,
                              prev.start, end, text[prev.start:end])
            continue
        out.append(Section(doc_type, doc_id, title, pos, end, body))
    return out


def record_sections(rec: Record) -> List[Section]:
    """레코드 전체를 절 목록으로. 공고문 먼저, 나머지는 DOC_ORDER 순."""
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    docs = sorted(rec.docs, key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))
    out: List[Section] = []
    for d in docs:
        out.extend(split_sections(d["text"], d["type"], d["doc_id"]))
    return out


# --------------------------------------------------------------------------- 항목군별 관련어

CUES: Dict[str, Sequence[str]] = {
    # G1 참가자격 제한 (v1~v8)
    "자격": (
        "입찰참가자격", "참가자격", "참가 자격", "자격요건", "참가등록", "등록증",
        "제한경쟁", "제한입찰", "실적", "이행실적", "납품실적", "수행실적", "시공능력",
        "주된 영업소", "본점소재지", "본사", "소재지", "관할구역", "지역제한", "소재한",
        "면허", "업종", "허가", "등록기준", "참여 가능", "참여가능", "한하여", "한함",
    ),
    # G2 기업규모·직접생산 (v10~v18)
    "기업규모": (
        "중소기업", "소기업", "소상공인", "중견기업", "대기업", "벤처기업", "창업기업",
        "직접생산", "직생", "직접생산확인증명서", "경쟁제품", "판로지원", "공공구매",
        "세부품명", "품명번호", "중소기업확인서", "확인서", "smpp",
    ),
    # G3 물품·SW·공동수급 (v9 v19 v20 v21)
    "물품SW공동": (
        "규격", "사양", "모델", "제조사", "제품명", "품목", "물품공급", "확약서", "협약서",
        "기술지원", "소프트웨어", "SW", "정보화", "시스템 구축", "대기업 참여", "참여제한",
        "공동수급", "공동계약", "공동도급", "공동이행", "분담이행", "지분", "출자비율", "구성원",
    ),
    # G4 설명회·대조 (v22 v23 v24)
    "설명회대조": (
        "현장설명회", "사업설명회", "제안요청 설명회", "설명회", "현장설명",
        "공고기간", "입찰공고", "게시일", "개찰", "제안서 제출", "입찰서 제출",
        "추정가격", "기초금액", "배정예산", "사업예산", "계약방법", "낙찰자 결정",
        "입찰에 부치는", "입찰개요",
    ),
}


def score_section(sec: Section, cues: Sequence[str]) -> int:
    """절의 관련도 = 신호어 출현 횟수(제목 가중 3배)."""
    n = 0
    for c in cues:
        if c in sec.title:
            n += 3
        n += sec.text.count(c)
    return n


def _emit(secs: List[Section], picked: List[int], budget: int) -> str:
    """선택된 절을 원문 순서로 잇는다.

    ⚠️ 인접한 절(인덱스 연속 + 같은 문서) 사이에는 아무것도 끼워 넣지 않는다.
       태그를 삽입하면 절 경계를 가로지르는 근거문구가 부분문자열 검사에서 탈락한다.
    """
    picked = sorted(set(picked))
    chunks: List[str] = []
    used = 0
    prev_idx: Optional[int] = None
    prev_doc: Optional[str] = None

    for i in picked:
        if used >= budget:
            break
        s = secs[i]
        tag = f"{s.doc_type}:{s.doc_id}"
        contiguous = (prev_idx is not None and i == prev_idx + 1 and tag == prev_doc)

        prefix = ""
        if not contiguous:
            prefix = ("\n\n" if chunks else "") + f"[{tag}]\n" if tag != prev_doc else "\n\n(…)\n"

        body = s.text
        room = budget - used - len(prefix)
        if room <= 0:
            break
        if len(body) > room:
            body = body[:room]
        chunks.append(prefix + body)
        used += len(prefix) + len(body)
        prev_idx, prev_doc = i, tag

    return "".join(chunks)


def select(
    rec: Record,
    cues: Sequence[str],
    budget: int,
    always_head: int = 1200,
) -> str:
    """관련도 높은 절부터 예산까지 채워 원문 순서대로 되돌려 반환한다.

    always_head: 공고문 첫머리(용역명·금액·계약방법 요약표·공고기간)는 거의 항상
                 판정에 쓰이므로 관련도와 무관하게 먼저 확보한다.
    """
    secs = record_sections(rec)
    if not secs:
        return ""

    picked: List[int] = []
    used = 0
    for i, s in enumerate(secs):
        if s.doc_type != "공고문":
            break
        if used >= always_head:
            break
        picked.append(i)
        used += len(s)

    ranked = sorted(
        (i for i in range(len(secs)) if i not in picked),
        key=lambda i: (-score_section(secs[i], cues), i),
    )
    for i in ranked:
        if score_section(secs[i], cues) <= 0:
            break
        if used >= budget:
            break
        picked.append(i)
        used += len(secs[i])

    return _emit(secs, picked, budget)


def full_context(rec: Record, budget: int) -> str:
    """부재탐지 항목용 — 전체 문서를 예산 안에서 최대한 담는다.

    잘렸으면 그 사실을 명시한다. '안 보이는 것'과 '없는 것'을 모델이 혼동하면
    부재탐지 항목이 통째로 틀린다.
    """
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    docs = sorted(rec.docs, key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))
    chunks: List[str] = []
    used = 0
    truncated: List[str] = []
    dropped: List[str] = []
    for d in docs:
        head = f"[{d['type']}:{d['doc_id']}]\n"
        body = d["text"]
        if used + len(head) + len(body) > budget:
            room = budget - used - len(head)
            if room > 300:
                chunks.append(head + body[:room])
                truncated.append(d["type"])
                used = budget
            else:
                dropped.append(d["type"])
            continue
        chunks.append(head + body)
        used += len(head) + len(body)

    text = "\n\n".join(chunks)
    notes = []
    if truncated:
        notes.append("길이 예산으로 뒷부분이 잘린 문서: " + ", ".join(sorted(set(truncated))))
    if dropped:
        notes.append("길이 예산으로 제외된 문서: " + ", ".join(sorted(set(dropped))))
    for t, n in (rec.dropped_doc_counts or {}).items():
        notes.append(f"애초에 제공되지 않은 문서: {t} {n}건")
    if notes:
        text += ("\n\n[주의] " + " / ".join(notes)
                 + "\n위 문서에 대해서는 '기재가 없다'고 단정하지 말 것.")
    return text
