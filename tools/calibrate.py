#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""예측률 보정 — 라벨 없이 분포 적합성을 검증한다.

왜 필요한가
  dev 200건은 README 가 명시하듯 평가 데이터 분포를 대표하지 않는다.
  실제로 train_unlabeled 20,000건과 견주면 dev 는 큰 계약에 편중돼 있다.
    판로지원 밴드 A(1억 미만): dev 53.0% vs 실제 71.7%
    밴드 C(고시금액 이상):      dev 22.5% vs 실제 11.2%
  그래서 dev 로 F1 을 올려도 리더보드가 따라오지 않는다(dev 0.63 → 리더보드 0.42).

  **예측률은 정답이 없어도 잴 수 있다.** 어떤 항목을 전체의 12% 비율로 위반이라
  찍는다면, 실제 위반율이 2% 인 이상 정밀도가 0.17 을 넘을 수 없다 — dev 와 무관하게
  증명된다. 이 도구는 그 상한을 계산한다.

사용:
    python3 tools/calibrate.py out/dev200b_pred.csv                    # dev 예측률
    python3 tools/calibrate.py out/holdout_pred.csv --true-rate 0.02   # 실제분포 표본
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import scoring  # noqa: E402
from pps.records import ITEMS, load_item_table  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")


def f1_ceiling(pred_rate: float, true_rate: float, recall: float = 1.0) -> float:
    """예측률과 실제 양성률이 주어졌을 때 도달 가능한 F1 상한.

    정밀도 = (재현율 × 실제양성률) / 예측률  (1.0 을 넘을 수 없다)
    """
    if pred_rate <= 0:
        return 0.0
    p = min(1.0, recall * true_rate / pred_rate)
    return 2 * p * recall / (p + recall) if (p + recall) else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("pred", help="submission 형식 예측 CSV")
    ap.add_argument("--labels", default=os.path.join(OPEN, "dev_labels.csv"),
                    help="있으면 실제 F1 도 함께 표시")
    ap.add_argument("--true-rate", type=float, default=0.02,
                    help="평가 데이터의 항목별 추정 양성률 (기본 2%%)")
    args = ap.parse_args()

    pred = scoring.load_submission(args.pred)
    n = len(pred)
    tbl = load_item_table(os.path.join(OPEN, "data"))

    gold = None
    if os.path.exists(args.labels):
        g = scoring.load_labels(args.labels)
        if set(pred) <= set(g):
            gold = {k: g[k] for k in pred}

    rep = scoring.score(pred, gold) if gold else None

    print(f"{args.pred} · {n}건 · 가정 실제 양성률 {args.true_rate:.1%}\n")
    header = f"{'항목':>5} {'예측률':>7} {'상한F1':>7}"
    if rep:
        header += f" {'실측F1':>7} {'P':>5} {'R':>5}"
    print(header + "  항목명")
    print("-" * (len(header) + 22))

    tot_pred = 0
    ceilings = []
    for it in ITEMS:
        c = sum(1 for r in pred.values() if r.get(it) == 1)
        tot_pred += c
        rate = c / n
        recall = rep.items[it].recall if rep else 1.0
        ceil = f1_ceiling(rate, args.true_rate, recall or 1.0)
        ceilings.append(ceil)
        line = f"{it:>5} {rate:>7.1%} {ceil:>7.3f}"
        if rep:
            s = rep.items[it]
            line += f" {(s.f1 or 0):>7.3f} {s.precision:>5.2f} {s.recall:>5.2f}"
        flag = "  ⚠️ 과예측" if rate > args.true_rate * 3 else ""
        print(line + f"  {tbl[it]['항목명'][:22]}{flag}")

    print("-" * (len(header) + 22))
    print(f"평균 예측률 {tot_pred / (n * 24):.2%} · "
          f"가정 양성률 {args.true_rate:.2%} · "
          f"과예측 배율 {tot_pred / (n * 24) / args.true_rate:.2f}x")
    print(f"예측률로 본 Macro F1 상한 = {sum(ceilings) / 24:.4f}")
    if rep:
        print(f"이 데이터에서의 실측 Macro F1 = {rep.macro_f1:.4f}")
    print("\n※ 상한은 '지금 예측률을 유지하면 아무리 잘 맞혀도 이 이상은 못 간다'는 뜻이다.")
    print("  상한이 목표보다 낮으면 정확도를 올리는 게 아니라 **예측을 줄여야** 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
