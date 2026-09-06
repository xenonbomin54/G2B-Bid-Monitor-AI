#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""섹션 파서 검증 — 추출한 텍스트가 정답 근거문구를 실제로 포함하는가.

이 지표가 프롬프트 입력 설계의 상한을 정한다.
추출본에 근거가 없으면 모델이 아무리 좋아도 그 항목은 맞힐 수 없다.

비교 대상
  baseline : 베이스라인 build_context (문서 순서대로 앞에서부터 자르기)
  sections : 항목군 신호어로 관련 절을 고른 뒤 원문 순서로 복원
  full     : 전체 문서를 예산까지 (부재탐지 항목용)

사용:
    python3 tools/probe_sections.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import scoring, sections  # noqa: E402
from pps.records import DOC_ORDER, load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")

# 항목 → 신호어 그룹
ITEM_CUE = {
    **{f"v{i}": "자격" for i in range(1, 9)},
    "v9": "물품SW공동",
    **{f"v{i}": "기업규모" for i in range(10, 19)},
    "v19": "물품SW공동", "v20": "물품SW공동", "v21": "물품SW공동",
    "v22": "설명회대조", "v23": "설명회대조", "v24": "설명회대조",
}


def baseline_context(rec, max_chars: int) -> str:
    """베이스라인 build_context 재현."""
    order = {t: i for i, t in enumerate(DOC_ORDER)}
    pool = sorted(rec.docs, key=lambda d: (order.get(d["type"], len(DOC_ORDER)), d["doc_id"]))
    chunks, used = [], 0
    for i, d in enumerate(pool):
        head = f"[{d['type']}:{d['doc_id']}]\n"
        body = d["text"]
        if used + len(head) + len(body) > max_chars:
            if i == 0:
                body = body[: max(0, max_chars - len(head))]
            else:
                continue
        chunks.append(head + body)
        used += len(head) + len(body)
    return "\n\n".join(chunks)


def main() -> int:
    recs = load_records(os.path.join(OPEN, "dev.jsonl.gz"))
    gold = scoring.load_labels(os.path.join(OPEN, "dev_labels.csv"))
    ev = scoring.load_evidence(os.path.join(OPEN, "dev_labels.csv"))

    # 인용 근거가 있는 (레코드, 항목) 쌍만 대상
    cases = []
    for r in recs:
        for i in range(1, 25):
            item = f"v{i}"
            e = ev[r.id][item]
            if e and gold[r.id][item] == 1:
                cases.append((r, item, e))
    print(f"인용 근거가 있는 정답 양성 {len(cases)}건\n")

    budgets = [3000, 4000, 6000, 8000, 12000]
    print(f"{'예산':>7} {'베이스라인':>12} {'섹션선택':>10} {'전체':>8}")
    print("-" * 42)
    for b in budgets:
        hit_base = hit_sec = hit_full = 0
        for r, item, e in cases:
            if e in baseline_context(r, b):
                hit_base += 1
            cues = sections.CUES[ITEM_CUE[item]]
            if e in sections.select(r, cues, b):
                hit_sec += 1
            if e in sections.full_context(r, b):
                hit_full += 1
        n = len(cases)
        print(f"{b:>7,} {hit_base / n:>11.1%} {hit_sec / n:>10.1%} {hit_full / n:>8.1%}")

    # 실패 사례 — 섹션 선택 4000자에서 놓친 것
    print("\n[섹션선택 4000자에서 놓친 근거]")
    miss = Counter()
    shown = 0
    for r, item, e in cases:
        cues = sections.CUES[ITEM_CUE[item]]
        if e not in sections.select(r, cues, 4000):
            miss[item] += 1
            if shown < 6:
                total = len(r.full_text)
                pos = r.full_text.find(e)
                print(f"  {r.id} {item}  문서{total:,}자 중 {pos:,}자 지점 "
                      f"({pos / total:.0%})  {e[:60]!r}")
                shown += 1
    print(f"  놓친 항목 분포: {dict(miss.most_common())}")

    # 압축률 — 같은 예산에서 얼마나 밀도 높은 입력을 주는가
    print("\n[입력 크기]")
    tot_full = sum(len(r.full_text) for r in recs)
    tot_sec = sum(len(sections.select(r, sections.CUES["자격"], 4000)) for r in recs)
    print(f"  원문 평균 {tot_full / len(recs):,.0f}자 → 섹션선택(자격,4000) 평균 "
          f"{tot_sec / len(recs):,.0f}자")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
