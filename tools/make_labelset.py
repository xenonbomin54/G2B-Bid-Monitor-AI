#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""사람이 라벨링할 검증셋 산출물 생성 — LLM·API 호출 없음.

## 왜 필요한가

dev 200건은 항목당 양성이 5~8건으로 **인위적으로 균형** 잡혀 있다(양성률 3.19%).
실제 평가셋은 그렇지 않다. 그 결과 정밀도가 실제에서 훨씬 나쁘게 나온다 —
리더보드에서 역산한 실제 정밀도는 0.331 인데 dev 는 0.498 이다(0.66배).
그래서 dev 로는 **FP 의 실제 비용을 측정할 수 없다.**

`out/holdout200.jsonl.gz` 는 train_unlabeled 20,000건에서 밴드·계약법·업무구분
비율을 유지해 뽑은 층화표본이다(tools/make_holdout.py). 라벨이 없다.
여기에 라벨을 붙이면 **실제 분포에서의 정밀도·FP 구성**을 직접 측정할 수 있다.

## 역할 분담 (중요)

    dev200      균형 표본 · 라벨 있음  → **재현율/FN** 측정에 쓴다
                (항목당 양성이 많아 재현율 추정이 안정적이다)
    holdout200  실제 분포 · 라벨 만든다 → **정밀도/FP** 측정에 쓴다

두 세트를 섞지 않는다. 앞으로 FP 감소 실험의 합격 조건은
"holdout200 에서 FP 감소 · dev200 에서 재현율 불변" 두 개를 **동시에** 만족하는 것이다.

## 이 도구가 만들지 않는 것

라벨을 만들지 않는다. `human_label`·`human_confidence`·`memo` 는 **빈 칸**으로 둔다.
모델 판정(`model_grade`·`model_pred`)은 참고용으로 나란히 싣지만,
사람이 그것을 베끼지 않도록 `human_label` 은 절대 미리 채우지 않는다.

원문은 **요약하지 않는다.** 근거문구 주변 원문을 그대로 잘라 넣고,
판정에 필요한 계산된 사실(등록정보·고시대조 등)은 파이프라인이 모델에게 준 것과
같은 내용을 그대로 싣는다. 사람이 모델과 같은 정보를 보고 독립적으로 판단해야
측정이 공정하다.

사용:
    # 예측 양성 전수 (정밀도 측정용) — grade 파일이 필요하다
    python3 tools/make_labelset.py --records out/holdout200.jsonl.gz \\
        --grades out/g_holdout.json --mode positives --out out/label_pos.csv

    # 레코드 단위 전수 (정밀도+재현율 측정용)
    python3 tools/make_labelset.py --records out/holdout200.jsonl.gz \\
        --grades out/g_holdout.json --mode records --sample 40 --out out/label_rec.csv

    # 예측 없이 레코드 표본만 (grade 파일 없을 때)
    python3 tools/make_labelset.py --records out/holdout200.jsonl.gz \\
        --mode records --sample 40 --out out/label_rec.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import gating, grades as gradeio, gosimatch, prompts, pumnum  # noqa: E402
from pps.prompts import GRADE_KEY  # noqa: E402
from pps.records import ABSENCE, ITEMS, load_item_table, load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")

COLS = [
    "record_id", "item", "item_name", "적용계약법", "업무구분", "추정가격", "밴드",
    "부재탐지", "model_grade", "model_pred",
    "evidence", "evidence_context", "판정기준", "계산된_사실",
    "human_label", "human_confidence", "memo",
]

CONTEXT = 700          # 근거문구 앞뒤로 보여줄 원문 글자 수
NO_EVIDENCE_HEAD = 1200  # 근거문구가 없을 때 보여줄 본문 앞부분


def context_around(text: str, quote: str | None, width: int = CONTEXT) -> str:
    """근거문구 주변 원문을 그대로 잘라 준다. 요약하지 않는다."""
    if not text:
        return ""
    if quote:
        key = re.sub(r"\s+", "", quote)[:40]
        flat = re.sub(r"\s+", "", text)
        pos = flat.find(key)
        if pos >= 0:
            # 압축 인덱스를 원문 인덱스로 되돌린다
            cnt = 0
            for i, ch in enumerate(text):
                if not ch.isspace():
                    if cnt == pos:
                        s = max(0, i - width)
                        return ("…" if s else "") + text[s: i + width] + "…"
                    cnt += 1
    return text[:NO_EVIDENCE_HEAD] + ("…" if len(text) > NO_EVIDENCE_HEAD else "")


def criteria_for(item: str) -> str:
    """그 항목의 판정 기준 — 파이프라인이 모델에게 준 것과 같은 문구."""
    grp = prompts.GROUP_OF[item]
    if grp.항목지시 and item in grp.항목지시:
        return grp.항목지시[item]
    return grp.지시


def facts_for(rec, item: str, tbl, gosi) -> str:
    """판정에 필요한 계산된 사실. 모델이 받은 것과 같은 내용만 싣는다."""
    parts = [prompts.fact_block(rec)]
    grp = prompts.GROUP_OF[item]
    try:
        if grp.key == "G2기업규모" and gosi is not None:
            parts.append("[중기부고시 경쟁제품 대조]\n" + gosimatch.block(rec, gosi))
        if item == "v9":
            from pps import spec
            parts.append("[모델명·제조사 후보]\n" + spec.candidate_block(rec))
        if item == "v21":
            from pps import compare
            parts.append("[공동수급 최소지분율(계산됨)]\n" + compare.joint_share_check(rec))
        if item == "v23":
            from pps import schedule
            parts.append("[일정 검산]\n" + schedule.fact_line(rec))
    except Exception as e:                                   # noqa: BLE001
        parts.append(f"(사실 계산 실패: {type(e).__name__})")
    return "\n\n".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", required=True, help="레코드 jsonl.gz")
    ap.add_argument("--grades", default=None,
                    help="원본 등급 json (없으면 model_grade/pred 는 빈칸)")
    ap.add_argument("--mode", choices=["positives", "records", "items"], default="positives",
                    help="positives=예측양성 전수 · records=레코드 단위 24항목 전수 · "
                         "items=특정 항목만")
    ap.add_argument("--items", default=None, help="mode=items 일 때 쉼표로 구분")
    ap.add_argument("--sample", type=int, default=None, help="mode=records 일 때 레코드 수")
    ap.add_argument("--threshold", type=int, default=2, help="등급 ≥ 이 값이면 예측 1")
    ap.add_argument("--seed", type=int, default=20260908)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    recs = load_records(args.records)
    by_id = {r.id: r for r in recs}
    tbl = load_item_table(os.path.join(OPEN, "data"))
    try:
        gosi = pumnum.load_gosi(os.path.join(OPEN, "data"))
    except Exception:                                        # noqa: BLE001
        gosi = None

    grades = {}
    if args.grades:
        grades, meta = gradeio.load(args.grades)
        print(f"등급 파일 {args.grades} · 메타 {meta}")

    # 대상 셀 선정
    targets: list[tuple[str, str]] = []
    if args.mode == "positives":
        if not grades:
            print("❌ mode=positives 는 --grades 가 필요하다"); return 2
        for rid, cells in grades.items():
            if rid not in by_id:
                continue
            g = gating.gate(by_id[rid])
            for it in ITEMS:
                if not g[it]:
                    continue
                if int((cells.get(it) or {}).get(GRADE_KEY, 0) or 0) >= args.threshold:
                    targets.append((rid, it))
    elif args.mode == "records":
        ids = sorted(by_id)
        if args.sample:
            random.Random(args.seed).shuffle(ids)
            ids = sorted(ids[: args.sample])
        for rid in ids:
            g = gating.gate(by_id[rid])
            targets += [(rid, it) for it in ITEMS if g[it]]
    else:
        want = [x.strip() for x in (args.items or "").split(",") if x.strip()]
        if not want:
            print("❌ mode=items 는 --items 가 필요하다"); return 2
        for rid in sorted(by_id):
            g = gating.gate(by_id[rid])
            targets += [(rid, it) for it in want if g[it]]

    rows = []
    for rid, it in targets:
        rec = by_id[rid]
        cell = (grades.get(rid) or {}).get(it) or {}
        grade = cell.get(GRADE_KEY)
        quote = cell.get("근거문구")
        pred = "" if grade is None else (1 if int(grade or 0) >= args.threshold else 0)
        rows.append({
            "record_id": rid,
            "item": it,
            "item_name": tbl.get(it, {}).get("항목명", ""),
            "적용계약법": rec.적용계약법,
            "업무구분": rec.업무구분,
            "추정가격": rec.추정가격,
            "밴드": rec.판로지원밴드,
            "부재탐지": "Y" if it in ABSENCE else "",
            "model_grade": "" if grade is None else grade,
            "model_pred": pred,
            "evidence": quote or "",
            "evidence_context": context_around(rec.full_text, quote),
            "판정기준": criteria_for(it),
            "계산된_사실": facts_for(rec, it, tbl, gosi),
            "human_label": "",          # ← 사람이 채운다. 절대 미리 채우지 않는다.
            "human_confidence": "",     # ← high / mid / low
            "memo": "",
        })

    # 레코드 단위로 묶어 정렬 — 같은 공고를 연달아 보게 해서 문맥 전환 비용을 줄인다
    rows.sort(key=lambda r: (r["record_id"], ITEMS.index(r["item"])))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with io.open(args.out, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)

    nrec = len({r["record_id"] for r in rows})
    npos = sum(1 for r in rows if r["model_pred"] == 1)
    print(f"✅ {args.out}")
    print(f"   라벨 대상 {len(rows):,}칸 · 공고 {nrec}건 · 모델 예측 양성 {npos}칸")
    print(f"   human_label 은 전부 빈칸이다 (모델이 대신 채우지 않는다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
