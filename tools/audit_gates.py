#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""게이팅 감사 — dev 라벨 기준으로 게이팅이 진짜 양성을 죽이는지 확인한다.

게이팅이 죽인 양성 1건 = 회복 불가능한 재현율 손실.
이 스크립트가 '치명'을 보고하면 해당 규칙은 법 해석이 틀린 것이므로 되돌려야 한다.

사용:
    python3 tools/audit_gates.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import gating, scoring  # noqa: E402
from pps.records import ITEMS, load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")


def main() -> int:
    recs = load_records(os.path.join(OPEN, "dev.jsonl.gz"))
    gold = scoring.load_labels(os.path.join(OPEN, "dev_labels.csv"))
    gates = {r.id: gating.gate(r) for r in recs}

    print(f"dev {len(recs)}건 · 라벨 {len(gold)}건\n")

    audits = scoring.audit_gates(gates, gold)
    print(scoring.format_gate_audit(audits, len(recs)))

    # 게이팅만 적용한 상한 점수 — "게이팅 통과 항목을 전부 1로 예측"했을 때의 재현율 상한
    ceiling = {rid: {i: (1 if g[i] else 0) for i in ITEMS} for rid, g in gates.items()}
    rep = scoring.score(ceiling, gold)
    print("\n[게이팅 상한] 통과 항목을 전부 1로 예측했을 때 (재현율 상한 확인용)")
    print(f"  Macro F1 {rep.macro_f1:.4f} · 평균 재현율 "
          f"{sum(s.recall for s in rep.items.values()) / 24:.4f}")

    # 치명 케이스 상세
    fatal = [a for a in audits if a.killed_positive]
    if fatal:
        print("\n[치명 케이스 상세]")
        by_id = {r.id: r for r in recs}
        for a in fatal:
            print(f"\n  {a.item} — 죽인 양성 {a.killed_positive}건 "
                  f"({gating.GATES[a.item].설명})")
            print(f"     근거: {gating.GATES[a.item].근거}")
            for rid, g in gold.items():
                if int(g.get(a.item, 0)) == 1 and not gates[rid][a.item]:
                    print("     " + gating.explain(by_id[rid]).replace("\n", "\n     "))
        return 1

    print("\n✅ 게이팅이 죽인 진짜 양성 없음 — 규칙 안전")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
