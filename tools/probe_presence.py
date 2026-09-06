#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""부재탐지 스캐너 검증 — 정규식 존재 판정이 dev 라벨과 얼마나 맞는가.

부재탐지 항목은 "요구 문구가 없음"이 위반이므로
    위반(=1)  ⟺  present == False  (그리고 적용 대상일 것)
이 성립해야 한다. 여기서는 게이팅까지 결합한 규칙 단독 성능을 본다.

사용:
    python3 tools/probe_presence.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import gating, presence, scoring  # noqa: E402
from pps.records import load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")
ABS_ITEMS = ["v10", "v11", "v16", "v18", "v20"]


def main() -> int:
    recs = load_records(os.path.join(OPEN, "dev.jsonl.gz"))
    gold = scoring.load_labels(os.path.join(OPEN, "dev_labels.csv"))

    scans = {r.id: presence.scan_record(r) for r in recs}
    gates = {r.id: gating.gate(r) for r in recs}

    print("부재탐지 항목 — 규칙 단독 성능 (게이팅 ∧ 문구부재)\n")
    print(f"{'항목':>5} {'정답+':>6} {'예측+':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'정밀도':>7} {'재현율':>7} {'F1':>7}")
    print("-" * 60)

    pred = {}
    for r in recs:
        sc = scans[r.id]
        g = gates[r.id]
        size = presence.size_restrictions(r.full_text)
        is_sw = sc["_SW사업"].present
        is_cp = sc["_경쟁제품"].present
        row = {
            # 경쟁제품 입찰인데 직접생산확인 요구가 없음
            "v10": 1 if (g["v10"] and is_cp and not sc["v10"].present) else 0,
            # 경쟁제품 입찰인데 중소기업자로 제한하지 않음
            "v11": 1 if (g["v11"] and is_cp and not size["중소기업"].present) else 0,
            # 1억~고시금액 구간인데 중소기업 제한이 없음
            "v16": 1 if (g["v16"] and not size["중소기업"].present) else 0,
            # 1억 미만 구간인데 소기업·소상공인 제한이 없음
            "v18": 1 if (g["v18"] and not size["소기업등"].present) else 0,
            # SW사업인데 대기업 참여제한 명시가 없음
            "v20": 1 if (g["v20"] and is_sw and not sc["v20"].present) else 0,
        }
        pred[r.id] = row

    rep = scoring.score(pred, gold)
    for item in ABS_ITEMS:
        s = rep.items[item]
        f1 = "  —  " if s.f1 is None else f"{s.f1:.4f}"
        print(f"{item:>5} {s.support:>6} {s.predicted:>6} {s.tp:>4} {s.fp:>4} {s.fn:>4} "
              f"{s.precision:>7.4f} {s.recall:>7.4f} {f1:>7}")
    avg = sum(rep.items[i].f1 or 0 for i in ABS_ITEMS) / len(ABS_ITEMS)
    print("-" * 60)
    print(f"부재탐지 5항목 평균 F1 = {avg:.4f}  (전체 Macro F1 기여 {avg * 5 / 24:.4f})")

    # 적용대상 신호 보조지표
    print("\n[적용대상 신호]")
    sw = sum(1 for r in recs if scans[r.id]["_SW사업"].present)
    cp = sum(1 for r in recs if scans[r.id]["_경쟁제품"].present)
    print(f"  SW사업 신호 감지 {sw}/{len(recs)}건 · v20 정답양성 {rep.items['v20'].support}건")
    print(f"  경쟁제품 신호 감지 {cp}/{len(recs)}건 · v10 정답양성 "
          f"{rep.items['v10'].support}건 · v11 {rep.items['v11'].support}건")

    # 오류 사례
    for item in ABS_ITEMS:
        fps = [r.id for r in recs if pred[r.id][item] == 1 and gold[r.id][item] == 0]
        fns = [r.id for r in recs if pred[r.id][item] == 0 and gold[r.id][item] == 1]
        if not (fps or fns):
            continue
        print(f"\n[{item}] FP {len(fps)} / FN {len(fns)}")
        for rid in fns[:3]:
            sc = scans[rid][item]
            print(f"  FN {rid}  게이팅통과={gates[rid][item]}  "
                  f"문구있음={sc.present} {sc.hits[:3]} sample={sc.sample!r}")
        for rid in fps[:3]:
            sc = scans[rid][item]
            print(f"  FP {rid}  문구있음={sc.present} {sc.hits[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
