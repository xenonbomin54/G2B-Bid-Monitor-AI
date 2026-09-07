#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""판정 문턱 스윕 — 저장된 LLM 원본 등급을 재채점한다. **API 호출 없음.**

왜 이 도구가 필요한가
  판정 문턱을 이진 0/1 로 받으면 문턱이 모델 안에 숨는다. 문턱 하나를 바꿔 보려면
  dev 200건 × 4그룹 = API 800건을 다시 호출해야 하고, 그 API 는 불안정해서
  85분이 걸리고 약 10%가 끊긴다. 등급(0~3)을 저장해 두면 호출은 한 번이고
  이후 비교는 전부 로컬이다.

무엇을 다시 계산하나
  저장물에는 **LLM 출력만** 있다. 게이팅·규칙·근거정합은 현재 코드로 다시 만든다
  (`Pipeline.finalize`). 그래서 규칙을 고친 뒤에도 같은 저장물로 재채점해
  효과를 견줄 수 있다 — 저장물은 LLM 호출 결과의 캐시이지 판정 결과가 아니다.

사용:
    python3 tools/evaluate.py --runner api --graded --save-grades out/g.json
    python3 tools/sweep_threshold.py out/g.json                # 문턱 1·2·3
    python3 tools/sweep_threshold.py out/g.json --per-item      # 항목별 최적 문턱까지
    python3 tools/sweep_threshold.py out/g.json --save-csv out/t2.csv --threshold 2
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import grades as gradeio  # noqa: E402
from pps import pumnum, scoring, submission  # noqa: E402
from pps.pipeline import GRADE_MIN_TH, GRADE_MAX_TH, Pipeline  # noqa: E402
from pps.records import ITEMS, load_item_table, load_records  # noqa: E402
from pps.runner import make_runner  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")


def f1(p: float, r: float) -> float:
    return 0.0 if p + r == 0 else 2 * p * r / (p + r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("grades", help="tools/evaluate.py --save-grades 로 만든 JSON")
    ap.add_argument("--data", default=os.path.join(OPEN, "dev.jsonl.gz"))
    ap.add_argument("--labels", default=os.path.join(OPEN, "dev_labels.csv"))
    ap.add_argument("--no-rules", action="store_true", help="규칙 결합 없이 LLM 단독")
    ap.add_argument("--per-item", action="store_true",
                    help="항목별로 최적 문턱을 찾아 그 조합의 Macro F1 까지 보고")
    ap.add_argument("--threshold", type=int, default=None,
                    help="이 문턱으로 제출 CSV 를 만든다 (--save-csv 와 함께)")
    ap.add_argument("--save-csv", default=None)
    args = ap.parse_args()

    grades, meta = gradeio.load(args.grades)
    recs = [r for r in load_records(args.data) if r.id in grades]
    gold_all = scoring.load_labels(args.labels)
    gold = {r.id: gold_all[r.id] for r in recs if r.id in gold_all}
    tbl = load_item_table(os.path.join(OPEN, "data"))
    gosi = pumnum.load_gosi(os.path.join(OPEN, "data"))

    print(f"등급 파일 {args.grades}")
    print(f"  메타 {meta}")
    print(f"  등급 분포 {gradeio.distribution(grades)}")
    print(f"  레코드 {len(recs)}건 · 라벨 {len(gold)}건 · **API 호출 없음**")
    if meta.get("판정모드") == "binary":
        print("  ⚠️ 이진 모드로 저장된 등급이다(0/3 만 있다) — 문턱 1~3 이 모두 같은 결과를 낸다.")

    # 러너는 만들지 않는다 — finalize 는 LLM 을 쓰지 않는다.
    pipe = Pipeline(None, tbl, gosi=gosi, use_rules=not args.no_rules)

    def score_at(th: int):
        pred = {}
        for r in recs:
            final = pipe.finalize(r, grades.get(r.id, {}), threshold=th)
            pred[r.id] = {v: final[v]["위반여부"] for v in ITEMS}
        return scoring.score(pred, gold), pred

    print(f"\n{'문턱':>4} {'예측+':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
          f"{'정밀도':>7} {'재현율':>7} {'MacroF1':>8}")
    print("-" * 52)
    best = None
    reports = {}
    for th in range(GRADE_MIN_TH, GRADE_MAX_TH + 1):
        rep, _ = score_at(th)
        reports[th] = rep
        TP = sum(s.tp for s in rep.items.values())
        FP = sum(s.fp for s in rep.items.values())
        FN = sum(s.fn for s in rep.items.values())
        P = TP / max(1, TP + FP)
        R = TP / max(1, TP + FN)
        print(f"{th:>4} {TP + FP:>6} {TP:>4} {FP:>4} {FN:>4} "
              f"{P:>7.3f} {R:>7.3f} {rep.macro_f1:>8.4f}")
        if best is None or rep.macro_f1 > best[1]:
            best = (th, rep.macro_f1)
    print("-" * 52)
    print(f"최적 단일 문턱 = {best[0]} (Macro F1 {best[1]:.4f})")

    if args.per_item:
        print("\n[항목별 문턱]  같은 저장물로 항목마다 문턱을 달리 준 경우")
        print(f"{'항목':>5} " + " ".join(f"th{t}:F1".rjust(9)
                                        for t in range(GRADE_MIN_TH, GRADE_MAX_TH + 1))
              + "   최적")
        total = 0.0
        for v in ITEMS:
            row, bth, bf1 = [], None, -1.0
            for th in range(GRADE_MIN_TH, GRADE_MAX_TH + 1):
                s = reports[th].items[v]
                row.append(f"{s.f1:9.3f}")
                if s.f1 > bf1:
                    bf1, bth = s.f1, th
            total += bf1
            print(f"{v:>5} " + " ".join(row) + f"   th{bth} {bf1:.3f}")
        print(f"\n항목별 최적 문턱 조합의 Macro F1 = {total / len(ITEMS):.4f}")
        print("⚠️ 이건 상한이다 — 같은 200건에서 항목마다 최적을 고른 것이라 과적합이다.")
        print("   항목당 양성이 5~8건뿐이라 ±0.15 흔들린다. 단일 문턱을 먼저 신뢰할 것.")

    if args.save_csv:
        th = args.threshold if args.threshold is not None else best[0]
        rows = [submission.to_row(r.id, pipe.finalize(r, grades.get(r.id, {}), threshold=th))
                for r in recs]
        submission.write_csv(rows, args.save_csv)
        errs = submission.validate(args.save_csv, [r.id for r in recs])
        print(f"\n문턱 {th} 로 저장 {args.save_csv} · 형식검증 "
              + ("PASS" if not errs else f"FAIL {errs[:3]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
