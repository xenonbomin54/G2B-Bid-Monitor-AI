#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""dev 200건 오프라인 평가 — 리더보드 산식(Macro F1) 재현.

1일 1회 제출 제약 때문에 이 스크립트가 사실상 리더보드 역할을 한다.
제출은 여기서 점수가 오른 경우에만 한다.

사용:
    python3 tools/evaluate.py --runner mock              # 배선 확인
    python3 tools/evaluate.py --runner api --limit 20    # API 소량
    python3 tools/evaluate.py --runner api               # dev 전체
    python3 tools/evaluate.py --runner api --save out/dev_pred.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import grades as gradeio  # noqa: E402
from pps import prompts, pumnum, scoring, submission  # noqa: E402
from pps.pipeline import GRADE_THRESHOLD_DEFAULT, Pipeline  # noqa: E402
from pps.records import ITEMS, load_item_table, load_records  # noqa: E402
from pps.runner import make_runner  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runner", default="mock", choices=["mock", "api", "vllm"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--save", default=None, help="예측을 submission 형식으로 저장")
    ap.add_argument("--no-rules", action="store_true", help="규칙 결합 끄고 LLM 단독 측정")
    ap.add_argument("--verify", action="store_true",
                    help="2단계 검증 — 1로 판정된 칸을 재질의해 과잉 판정을 걷어낸다")
    ap.add_argument("--data", default=os.path.join(OPEN, "dev.jsonl.gz"))
    ap.add_argument("--labels", default=os.path.join(OPEN, "dev_labels.csv"))
    # 기본을 등급 모드로 둔다 — build_submit.py 의 제출 기본값과 **같아야** 한다.
    # 측정 모드와 제출 모드가 갈리면 측정값이 제출을 설명하지 못한다(그 함정에 한 번 빠졌다).
    ap.add_argument("--graded", dest="graded", action="store_true", default=True,
                    help="LLM 에 위반등급(0~3)을 요구한다. **기본 켜짐**(제출 설정과 동일). "
                         "저장해 두면 tools/sweep_threshold.py 로 API 없이 문턱을 스윕할 수 있다.")
    ap.add_argument("--binary", dest="graded", action="store_false",
                    help="위반여부 0/1 로 받는다. 제출 설정과 달라지므로 비교용으로만.")
    ap.add_argument("--grade-threshold", type=int, default=GRADE_THRESHOLD_DEFAULT,
                    help=f"등급 ≥ 이 값이면 위반 (기본 {GRADE_THRESHOLD_DEFAULT})")
    ap.add_argument("--save-grades", default=None,
                    help="LLM 원본 등급을 JSON 으로 저장한다 (문턱 스윕용)")
    args = ap.parse_args()

    recs = load_records(args.data, limit=args.limit)
    gold_all = scoring.load_labels(args.labels)
    gold = {r.id: gold_all[r.id] for r in recs if r.id in gold_all}
    tbl = load_item_table(os.path.join(OPEN, "data"))
    gosi = pumnum.load_gosi(os.path.join(OPEN, "data"))

    print(f"레코드 {len(recs)}건 · 러너 {args.runner}"
          f"{' · 규칙결합 OFF' if args.no_rules else ''}"
          f"{' · 등급모드(0~3)' if args.graded else ''}"
          f" · 문턱 {args.grade_threshold}")

    runner = make_runner(args.runner, items=ITEMS)
    pipe = Pipeline(runner, tbl, gosi=gosi, use_rules=not args.no_rules,
                    graded=args.graded, grade_threshold=args.grade_threshold)

    judged = pipe.run(recs, chunk=args.chunk)

    # LLM 원본 등급을 먼저 저장한다 — 이후 문턱 스윕은 API 없이 이 파일로 한다.
    if args.save_grades:
        gradeio.save(args.save_grades, judged, meta={
            "판정모드": "graded" if args.graded else "binary",
            "문턱": args.grade_threshold,
            "러너": args.runner,
            "데이터": os.path.basename(args.data),
            "시드": pipe.seed,
            "max_tokens": pipe.max_tokens,
        })
        dist = gradeio.distribution(judged)
        print(f"  원본 등급 저장 {args.save_grades} · 등급분포 {dist}")

    dropped = {}
    if args.verify:
        dropped = pipe.verify(recs, judged, chunk=args.chunk)

    rows = []
    pred = {}
    for r in recs:
        final = pipe.finalize(r, judged.get(r.id, {}), dropped.get(r.id))
        rows.append(submission.to_row(r.id, final))
        pred[r.id] = {v: final[v]["위반여부"] for v in ITEMS}

    print("\n" + pipe.stats.report())

    rep = scoring.score(pred, gold)
    print("\n" + rep.table())

    print("\n[개선 우선순위 — F1 낮은 항목]")
    for s in rep.weakest(8):
        name = tbl.get(s.item, {}).get("항목명", "")
        print(f"  {s.item:>4} F1 {s.f1:.4f}  P {s.precision:.3f} R {s.recall:.3f}  "
              f"(정답+{s.support} 예측+{s.predicted})  {name}")

    # 근거문구 원문 정합 확인 (2차 평가 대비)
    sources = {r.id: r.full_text for r in recs}
    bad = submission.verify_evidence_in_source(rows, sources)
    n_ev = sum(1 for row in rows for i in range(1, 25) if row[f"e{i}"])
    print(f"\n[근거문구] 제출 {n_ev}개 · 원문 불일치 {len(bad)}개")
    for b in bad[:5]:
        print("  " + b)

    if args.save:
        submission.write_csv(rows, args.save)
        errs = submission.validate(args.save, [r.id for r in recs])
        print(f"\n저장 {args.save} · 형식검증 "
              + ("PASS" if not errs else f"FAIL {errs[:3]}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
