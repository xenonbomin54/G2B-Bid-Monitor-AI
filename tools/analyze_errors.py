#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""오답 분석 — F1이 낮은 항목의 실패 원인을 층으로 갈라서 보여준다.

파이프라인은 층이 여러 개라 "F1이 낮다"만으로는 어디를 고쳐야 할지 알 수 없다.
실패를 원인별로 나눈다.

  게이팅차단   정답은 1인데 게이팅이 0으로 막음        → 법 해석 오류. 최우선.
  입력누락     근거문구가 프롬프트에 안 들어감          → 섹션 파서·예산 문제
  규칙보정     LLM은 1이라 했는데 규칙이 0으로 내림     → 규칙 힌트 오류
  모델오판     입력에 근거가 있는데 LLM이 0/1을 틀림     → 프롬프트 문제
  호출실패     빈 출력·결손                            → 파싱·타임아웃 문제

사용:
    python3 tools/analyze_errors.py out/dev20_pred.csv
    python3 tools/analyze_errors.py out/dev_pred.csv --item v13 --show 5
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from typing import Dict, List

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import gating, presence, prompts, scoring, sections  # noqa: E402
from pps.records import ITEMS, load_item_table, load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")

ITEM_CUE = {
    **{f"v{i}": "자격" for i in range(1, 9)},
    "v9": "물품SW공동",
    **{f"v{i}": "기업규모" for i in range(10, 19)},
    "v19": "물품SW공동", "v20": "물품SW공동", "v21": "물품SW공동",
    "v22": "설명회대조", "v23": "설명회대조", "v24": "설명회대조",
}


def prompt_text(rec, item: str) -> str:
    """해당 항목이 실제로 들어간 프롬프트의 문서 부분을 재현한다."""
    g = prompts.GROUP_OF[item]
    if g.full_doc:
        return sections.full_context(rec, g.budget)
    return sections.select(rec, sections.CUES[g.cue], g.budget)


def classify_fn(rec, item: str, gold_ev: str) -> str:
    """정답 1을 놓친 원인."""
    if not gating.gate(rec)[item]:
        return "게이팅차단"
    if gold_ev and gold_ev not in prompt_text(rec, item):
        return "입력누락"
    return "모델오판"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="submission 형식 예측 CSV")
    ap.add_argument("--item", default=None, help="특정 항목만")
    ap.add_argument("--show", type=int, default=3, help="항목당 사례 수")
    ap.add_argument("--data", default=os.path.join(OPEN, "dev.jsonl.gz"))
    ap.add_argument("--labels", default=os.path.join(OPEN, "dev_labels.csv"))
    args = ap.parse_args()

    pred = scoring.load_submission(args.pred)
    recs = [r for r in load_records(args.data) if r.id in pred]
    by_id = {r.id: r for r in recs}
    gold_all = scoring.load_labels(args.labels)
    ev_all = scoring.load_evidence(args.labels)
    gold = {r.id: gold_all[r.id] for r in recs}
    tbl = load_item_table(os.path.join(OPEN, "data"))

    rep = scoring.score(pred, gold)
    print(f"예측 {args.pred} · 레코드 {len(recs)}건")
    print(f"Macro F1 = {rep.macro_f1:.4f}\n")

    items = [args.item] if args.item else ITEMS
    causes = Counter()

    for item in items:
        s = rep.items[item]
        if s.f1 is None:
            continue
        fns = [r for r in recs if gold[r.id][item] == 1 and pred[r.id][item] == 0]
        fps = [r for r in recs if gold[r.id][item] == 0 and pred[r.id][item] == 1]
        if not (fns or fps):
            continue

        name = tbl.get(item, {}).get("항목명", "")
        print("=" * 78)
        print(f"{item}  F1 {s.f1:.3f}  P {s.precision:.3f} R {s.recall:.3f}  "
              f"(정답+{s.support} 예측+{s.predicted})  {name}")

        if fns:
            cs = Counter(classify_fn(by_id[r.id], item, ev_all[r.id][item]) for r in fns)
            causes.update({f"{item}/{k}": v for k, v in cs.items()})
            print(f"\n  놓침 {len(fns)}건 — 원인: {dict(cs)}")
            for r in fns[:args.show]:
                cause = classify_fn(r, item, ev_all[r.id][item])
                gev = ev_all[r.id][item]
                print(f"    [{cause}] {r.id} · {r.적용계약법} {r.업무구분} "
                      f"{r.추정가격:,}원 밴드{r.판로지원밴드}")
                if gev:
                    print(f"        정답근거: {gev[:110]!r}")
                if cause == "게이팅차단":
                    print(f"        ⚠️ {gating.GATES[item].근거}")

        if fps:
            print(f"\n  헛짚음 {len(fps)}건")
            for r in fps[:args.show]:
                print(f"    {r.id} · {r.적용계약법} {r.업무구분} "
                      f"{r.추정가격:,}원 밴드{r.판로지원밴드}"
                      + (f" · 조항호={r.meta.get('조항호내용')}"
                         if r.meta.get("조항호내용") else ""))
        print()

    if not args.item:
        print("=" * 78)
        print("전체 원인 분포 (놓침 기준)")
        agg = Counter()
        for k, v in causes.items():
            agg[k.split("/")[1]] += v
        for k, v in agg.most_common():
            print(f"  {k:>8} {v:>4}건")
        print("\n권고: '게이팅차단'이 있으면 법 해석부터 고친다 — 회복 불가능한 손실이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
