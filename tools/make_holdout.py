#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_unlabeled 20,000건에서 실제 분포를 유지한 층화표본을 뽑는다.

왜 필요한가
  dev 200건은 README 가 명시하듯 평가 데이터 분포를 대표하지 않는다.
  실측하면 dev 는 큰 계약에 편중돼 있다 — 판로지원 밴드 A(1억 미만)가
  dev 53.0% vs 실제 71.7%, 밴드 C(고시금액 이상)가 dev 22.5% vs 실제 11.2%.
  그래서 dev F1 을 올려도 리더보드가 따라오지 않았다(0.6326 → 0.42458).

층화 기준: 판로지원 밴드 × 적용계약법 × 업무구분
  이 세 축이 게이팅을 결정하므로, 비율이 어긋나면 항목별 적용률이 달라진다.

시드를 고정해 재현 가능하게 한다(규칙 3 재현성 요구).

사용:
    python3 tools/make_holdout.py                    # 200건, out/holdout200.jsonl.gz
    python3 tools/make_holdout.py -n 500 --seed 1
"""
from __future__ import annotations

import argparse
import collections
import gzip
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps.records import _to_record, validate  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "open", "train_unlabeled.jsonl.gz")


def strata(r):
    return (r.판로지원밴드, r.적용계약법, r.업무구분)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20260906)
    ap.add_argument("--out", default=os.path.join(ROOT, "out", "holdout200.jsonl.gz"))
    args = ap.parse_args()

    random.seed(args.seed)
    recs = []
    with gzip.open(SRC, "rt", encoding="utf-8") as f:
        for line in f:
            raw = json.loads(line)
            validate(raw)
            recs.append(_to_record(raw))
    print(f"모집단 {len(recs):,}건")

    buckets: dict = {}
    for r in recs:
        buckets.setdefault(strata(r), []).append(r)

    out = []
    for k, v in sorted(buckets.items(), key=lambda x: -len(x[1])):
        take = max(1, round(args.n * len(v) / len(recs)))
        out.extend(random.sample(v, min(take, len(v))))
    out = out[: args.n]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with gzip.open(args.out, "wt", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r.raw, ensure_ascii=False) + "\n")

    print(f"표본 {len(out)}건 → {args.out}\n")
    for name, key in [("판로지원밴드", lambda r: r.판로지원밴드),
                      ("적용계약법", lambda r: r.적용계약법),
                      ("업무구분", lambda r: r.업무구분)]:
        s = collections.Counter(key(r) for r in out)
        p = collections.Counter(key(r) for r in recs)
        parts = [f"{k} {s[k] / len(out):.0%}(모집단 {p[k] / len(recs):.0%})"
                 for k in sorted(s)]
        print(f"  {name}: " + " · ".join(parts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
