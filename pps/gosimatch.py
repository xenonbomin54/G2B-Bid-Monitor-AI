# -*- coding: utf-8 -*-
"""조달 대상 ↔ 중기부고시 경쟁제품 유사도 매칭 (v10·v11·v12·v13).

왜 필요한가 — dev 200건에서 확인한 사실
  · v10∪v11∪v13 양성 15건은 **전부 일반용역**이고 물품은 0건이다.
  · 공고문이 "중소기업자간 경쟁제품"이라고 **밝힌** 14건은 양성과 겹침이 **0건**이다.
    즉 스스로 인지한 기관은 직생·중소 요건도 제대로 걸어서 위반이 아니다.
    위반은 **경쟁제품인데 그 사실이 공고문에 드러나지 않은** 건이다.
    → 공고문 문구로는 절대 못 잡는다. **조달 대상 품목 자체**로 판별해야 한다.
  · 고시 세부품명은 "기타행사기획및대행서비스"인데 공고는 "라이브공연 행사 대행 용역"
    이라고 쓴다. 완전일치가 실패하는 이유다.

방법
  용역명·물품명을 뽑아 고시 615개 품명과 **문자 바이그램 자카드 유사도**로 견준다.
  형태소 분석 없이도 한국어 부분 일치를 잘 잡고, 외부 모델이 필요 없다.
  (평가 서버에는 bge-m3 가 있으므로 임베딩 매칭으로 올릴 여지가 있다 — 다만
   고정 LLM 과 GPU 동시 적재가 보장되지 않아 우선 규칙으로 해결한다.)

이 모듈은 **판정하지 않는다.** 후보와 점수를 내놓고 최종 판단은 LLM 이 한다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .pumnum import Gosi, GosiItem
from .records import Record

# 품명에서 걷어낼 군더더기 — 있으나 없으나 뜻이 같다
_STRIP = re.compile(
    r"(주식회사|㈜|기타|및|등|외|용역|사업|공고|입찰|구매|구입|제작|납품|설치|"
    r"서비스|업무|일식|건|년도|년|차|호|제\d+)")
_NONWORD = re.compile(r"[^가-힣A-Za-z0-9]")

# 공고 제목·품명을 뽑는 자리
_NAME_PAT = re.compile(
    r"(용\s*역\s*명|물\s*품\s*명|사\s*업\s*명|건\s*명|입\s*찰\s*명|공\s*고\s*명|과\s*업\s*명)"
    r"\s*[:：]?\s*([^\n|]{4,80})")


def norm(s: str) -> str:
    s = _STRIP.sub("", s)
    return _NONWORD.sub("", s)


def bigrams(s: str) -> set:
    s = norm(s)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def containment(a: set, b: set) -> float:
    """a 가 b 에 얼마나 담기는가. 고시 품명이 짧고 공고명이 길 때 자카드보다 낫다."""
    if not a:
        return 0.0
    return len(a & b) / len(a)


def procurement_names(rec: Record, limit: int = 6) -> List[str]:
    """이 공고가 무엇을 조달하는지 나타내는 문구들."""
    out: List[str] = []
    seen = set()
    head = rec.notice_text[:3000]
    for m in _NAME_PAT.finditer(head):
        v = re.sub(r"\s+", " ", m.group(2)).strip(" .:-|")
        if len(v) >= 4 and v not in seen:
            seen.add(v)
            out.append(v)
    # 메타 세부품명 이름 부분 — '잡지[5510150601]' 의 '잡지'
    s = str(rec.meta.get("세부품명번호목록") or "")
    for m in re.finditer(r"([가-힣A-Za-z0-9 ]{2,30})\[\d{10}\]", s):
        v = m.group(1).strip()
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    if not out:                      # 이름 자리를 못 찾으면 공고문 첫 줄
        first = re.sub(r"\s+", " ", rec.notice_text[:200]).strip()
        out.append(first[:80])
    return out[:limit]


@dataclass
class GosiHit:
    item: GosiItem
    score: float
    matched_name: str

    def line(self) -> str:
        note = f" · 특이사항: {self.item.특이사항}" if self.item.특이사항 else ""
        return (f"- {self.item.세부품명번호} {self.item.세부품명} "
                f"(분류: {self.item.제품명}) · 유사도 {self.score:.2f} "
                f"← 공고의 '{self.matched_name}'{note}")


def match(rec: Record, gosi: Gosi, topk: int = 5,
          min_score: float = 0.30) -> List[GosiHit]:
    """조달 대상과 가장 비슷한 고시 품목."""
    names = procurement_names(rec)
    name_bg = [(n, bigrams(n)) for n in names]
    best: dict = {}
    for it in gosi.by_num.values():
        gb = bigrams(it.세부품명)
        if not gb:
            continue
        for n, nb in name_bg:
            if not nb:
                continue
            # 고시 품명이 공고명 안에 얼마나 담기는가 + 전체 자카드
            s = max(containment(gb, nb), jaccard(gb, nb))
            prev = best.get(it.세부품명번호)
            if prev is None or s > prev[0]:
                best[it.세부품명번호] = (s, n)
    hits = [GosiHit(gosi.by_num[k], s, n) for k, (s, n) in best.items()
            if s >= min_score]
    hits.sort(key=lambda h: -h.score)
    return hits[:topk]


def exact_numbers(rec: Record, gosi: Gosi) -> List[str]:
    """메타·본문에 적힌 10자리 세부품명번호 중 고시에 있는 것 (확실한 신호)."""
    nums = set(rec.세부품명번호) | set(re.findall(r"\d{10}", rec.full_text))
    return sorted(nums & gosi.numbers)


# 고시 615개 중 '용역·서비스'로 조달되는 품목.
# dev 200건에서 v10·v11·v13 양성 15건이 **전부 일반용역**이었으므로 이 목록이 핵심이다.
# 40여 개뿐이라 프롬프트에 통째로 넣을 수 있다 — 문자열 유사도로 흉내내는 것보다
# LLM 의 의미 매칭이 훨씬 정확하다(바이그램 매칭은 '통학차량 임차'를 '교통신호등'에,
# '전시연출'을 '전시대'에 붙였다. 정밀도 0.14).
# 품명이 서비스/용역으로 끝나거나 대행·기획 업무인 것만. '청소도구함'·'경비선' 같은
# 물품이 섞이지 않도록 어미를 기준으로 잡는다.
_SERVICE_PAT = re.compile(r"(서비스|용역)$|대행서비스|기획및")


def service_catalog(gosi: Gosi) -> List[GosiItem]:
    out = [it for it in gosi.by_num.values() if _SERVICE_PAT.search(it.세부품명)]
    out.sort(key=lambda i: i.세부품명번호)
    return out


def _catalog_lines(items: Sequence[GosiItem], limit: int = 60) -> List[str]:
    return [f"- {i.세부품명번호} {i.세부품명} (분류: {i.제품명})" for i in items[:limit]]


def block(rec: Record, gosi: Gosi) -> str:
    """G2 프롬프트용 — 조달 대상과 고시 후보 목록.

    판정하지 않는다. 후보를 좁혀 주고 의미 매칭은 LLM 이 한다.
    """
    names = procurement_names(rec)
    lines = ["이 공고의 조달 대상: " + " / ".join(names[:3])]

    ex = exact_numbers(rec, gosi)
    if ex:
        # ⚠️ 예전에는 "(확실한 근거)" 라고 단정했다. 그러면 '코드가 고시에 있다'는 사실이
        #    곧 '이 조달은 경쟁제품이다'로 읽혀 v12 판정을 막는다. dev200i 의 v12 놓침
        #    PPS-DEV-053·056 이 그 사례다 — 발주기관이 조달 대상과 무관한 코드를 붙여
        #    직접생산확인을 요구했는데(그것이 곧 v12 위반), 블록이 적법처럼 안내했다.
        #      053 조달대상 '알츠하이머 후보소재 비교·분석' vs 코드 '행사기획/전시회기획'
        #      056 조달대상 '해운부문 감축사업 활성화 용역' vs 코드 '인터넷지원개발서비스'
        #    사실(코드·명칭·조달대상)만 주고 일치 여부는 모델이 판단한다.
        #    ⚠️ 어느 쪽으로도 기울이지 않는다 — 일치하면 경쟁제품이 맞고, 그때는
        #      v10·v11·v13 판정이 그대로 서야 한다.
        lines.append("문서에 적혀 있는 고시 세부품명번호:")
        for n in ex[:6]:
            it = gosi.by_num[n]
            lines.append(f"- {n} {it.세부품명} (분류: {it.제품명})")
        lines.append(
            "  ※ **코드가 적혀 있다는 사실과, 그 코드가 이 조달 대상에 맞는 코드라는 사실은 다르다.**"
            " 위 '조달 대상'과 이 코드의 명칭을 견주어라."
            " 서로 맞으면 이 입찰은 경쟁제품 입찰이다."
            " 조달 대상과 무관한 코드를 붙여 놓은 것이라면 그 코드는 경쟁제품 근거가 되지 못한다"
            " — 그런데도 직접생산확인을 요구했다면 그것이 v12 다.")

    if not rec.is_물품:
        cat = service_catalog(gosi)
        lines.append(f"\n고시에 등재된 **용역·서비스** 경쟁제품 전체 목록 ({len(cat)}개) — "
                     f"이 공고가 조달하는 용역이 아래 중 하나에 해당하는지 판단하라:")
        lines.extend(_catalog_lines(cat))
    else:
        hits = match(rec, gosi)
        if hits:
            lines.append("\n품명이 비슷한 고시 물품 후보(문자 유사도이므로 뜻이 다를 수 있다. "
                         "실제로 같은 품목인지 직접 판단하라):")
            lines.extend(h.line() for h in hits)
        elif not ex:
            lines.append("\n고시 물품 목록과 비슷한 항목을 찾지 못했다.")

    lines.append(
        "\n※ 공고문이 '중소기업자간 경쟁제품'이라고 **밝히지 않았더라도**, "
        "조달 대상이 고시 품목에 해당하면 경쟁제품 입찰이다. "
        "오히려 그 사실을 밝히지 않은 채 직접생산확인·중소기업자 제한을 걸지 않은 것이 "
        "전형적인 위반 유형이다. 공고문의 자기 진술이 아니라 **무엇을 사는지**로 판단하라.")
    return "\n".join(lines)
