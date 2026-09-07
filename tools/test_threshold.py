#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문턱 스윕 구조 자체 검증 — LLM·API 호출 없음.

검사 항목
  1. 기존 v24 회귀      — 이진 모드 판정이 예전과 같은가 (규칙 미배선 상태 유지)
  2. 등급 파싱          — 0~3 · 이진 · 축약형 · 결손 · 범위초과
  3. 문턱 판정          — 등급 ≥ 문턱만 1 이 되는가, 게이팅/규칙보다 뒤인가
  4. 문턱 스윕          — 문턱을 올리면 예측 수가 단조 감소하는가
  5. 저장/재로딩        — 원본 등급이 손실 없이 왕복하는가
  6. API 무호출         — 스윕 경로가 러너를 건드리지 않는가 (러너 None 으로 확인)
  7. 제출 형식          — 49열·0/1·근거문구 규약을 지키는가

사용:
    python3 tools/test_threshold.py
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps import grades as gradeio  # noqa: E402
from pps import gating, prompts, submission  # noqa: E402
from pps.pipeline import (GRADE_MAX_TH, GRADE_MIN_TH, GRADE_THRESHOLD_DEFAULT,  # noqa: E402
                          Pipeline, parse_group)
from pps.records import ABSENCE, COLUMNS, ITEMS, load_item_table, load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")
G = prompts.GRADE_KEY

_fails = []


def check(name: str, cond: bool, detail: str = "") -> None:
    print(("  OK   " if cond else "  FAIL ") + name + (f"  — {detail}" if detail else ""))
    if not cond:
        _fails.append(name)


# --------------------------------------------------------------------------- 2. 파싱
def test_parse():
    print("\n[2] 등급 파싱")
    items = ["v1", "v2", "v3"]

    out, miss = parse_group(
        '{"v1":{"위반등급":3,"근거문구":"가"},'
        '"v2":{"위반등급":1,"근거문구":"나"},'
        '"v3":{"위반등급":0,"근거문구":null}}', items, graded=True)
    check("등급 0~3 을 그대로 읽는다",
          [out[v][G] for v in items] == [3, 1, 0], str([out[v][G] for v in items]))
    check("근거문구를 보존한다", out["v1"]["근거문구"] == "가")
    check("결손 없음", miss == [])

    out, _ = parse_group('{"v1":{"위반여부":1,"근거문구":"가"},'
                         '"v2":{"위반여부":0,"근거문구":null},'
                         '"v3":{"위반여부":1,"근거문구":"다"}}', items, graded=False)
    check("이진 1 은 GRADE_MAX 로, 0 은 0 으로 담긴다",
          [out[v][G] for v in items] == [prompts.GRADE_MAX, 0, prompts.GRADE_MAX],
          str([out[v][G] for v in items]))

    out, _ = parse_group('{"v1":2,"v2":"1","v3":0}', items, graded=True)
    check("축약형(스칼라)도 살린다",
          [out[v][G] for v in items] == [2, 1, 0], str([out[v][G] for v in items]))

    out, miss = parse_group('{"v1":{"위반등급":9,"근거문구":null},'
                            '"v2":{"위반등급":-4,"근거문구":null}}', items, graded=True)
    check("범위를 벗어난 등급을 0~3 으로 잘라 맞춘다",
          out["v1"][G] == prompts.GRADE_MAX and out["v2"][G] == 0)
    check("빠진 항목은 결손으로 보고하고 0 으로 채운다",
          miss == ["v3"] and out["v3"][G] == 0, str(miss))

    out, miss = parse_group("완전히 깨진 출력", items, graded=True)
    check("파싱 실패 시 전 항목 0 · 결손 보고",
          all(out[v][G] == 0 for v in items) and miss == items)

    # 모드를 헷갈려 다른 키로 낸 경우
    out, _ = parse_group('{"v1":{"위반여부":1,"근거문구":null}}', ["v1"], graded=True)
    check("등급 모드인데 위반여부로 온 것도 받는다", out["v1"][G] == prompts.GRADE_MAX)


# --------------------------------------------------------------------------- 3·4. 문턱
def _pipe(recs, tbl):
    # 러너를 None 으로 준다 — finalize 는 LLM 을 쓰지 않는다(6번 검사와 같은 근거).
    return Pipeline(None, tbl)


def test_threshold(recs, tbl):
    print("\n[3] 문턱 판정")
    pipe = _pipe(recs, tbl)
    rec = recs[0]
    g = gating.gate(rec)
    open_items = [v for v in ITEMS if g[v]]
    check("게이팅 통과 항목이 있다", bool(open_items), str(len(open_items)))

    v = open_items[0]
    for grade in range(0, prompts.GRADE_MAX + 1):
        judged = {v: {G: grade, "근거문구": None}}
        for th in range(GRADE_MIN_TH, GRADE_MAX_TH + 1):
            hit = pipe.finalize(rec, judged, threshold=th)[v]["위반여부"]
            want = 1 if grade >= th else 0
            if hit != want:
                check(f"등급{grade} 문턱{th} → {want}", False, f"실제 {hit}")
                return
    check("등급 ≥ 문턱일 때만 1 이 된다 (전 조합)", True)

    # 게이팅이 문턱보다 앞선다
    blocked = [v for v in ITEMS if not g[v]]
    if blocked:
        b = blocked[0]
        judged = {b: {G: prompts.GRADE_MAX, "근거문구": None}}
        out = pipe.finalize(rec, judged, threshold=1)
        check("게이팅으로 막힌 항목은 최고 등급이어도 0", out[b]["위반여부"] == 0)

    # threshold 인자를 주지 않으면 설정값을 쓴다
    judged = {v: {G: GRADE_THRESHOLD_DEFAULT, "근거문구": None}}
    check("threshold 를 안 주면 GRADE_THRESHOLD_DEFAULT 를 쓴다",
          pipe.finalize(rec, judged)[v]["위반여부"] == 1)

    # 원본 등급이 변형되지 않는다
    judged = {v: {G: 2, "근거문구": "x"}}
    pipe.finalize(rec, judged, threshold=3)
    check("finalize 가 원본 등급을 건드리지 않는다", judged[v][G] == 2)


def test_sweep_monotonic(recs, tbl):
    print("\n[4] 문턱 스윕 — 단조성")
    pipe = _pipe(recs, tbl)
    # 등급을 항목별로 흩어 준다 (0,1,2,3 순환)
    judged_all = {}
    for i, r in enumerate(recs):
        judged_all[r.id] = {v: {G: (i + j) % 4, "근거문구": "근거"}
                            for j, v in enumerate(ITEMS)}
    counts = []
    for th in range(GRADE_MIN_TH, GRADE_MAX_TH + 1):
        n = sum(pipe.finalize(r, judged_all[r.id], threshold=th)[v]["위반여부"]
                for r in recs for v in ITEMS)
        counts.append(n)
    print(f"       문턱별 예측+ 수: {counts}")
    check("문턱을 올리면 예측 수가 줄어든다(비증가)",
          all(counts[i] >= counts[i + 1] for i in range(len(counts) - 1)), str(counts))
    check("문턱 1 과 3 의 예측 수가 실제로 다르다", counts[0] != counts[-1], str(counts))


# --------------------------------------------------------------------------- 5. 왕복
def test_roundtrip(recs):
    print("\n[5] 저장 / 재로딩")
    src = {}
    for i, r in enumerate(recs[:5]):
        src[r.id] = {v: {G: (i + j) % 4,
                         "근거문구": (None if v in ABSENCE else f"근거 {v}")}
                     for j, v in enumerate(ITEMS)}
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "g.json")
        gradeio.save(p, src, meta={"판정모드": "graded", "문턱": 2})
        got, meta = gradeio.load(p)
        check("레코드 수 왕복", set(got) == set(src))
        same = all(got[rid][v][G] == src[rid][v][G] and
                   got[rid][v]["근거문구"] == src[rid][v]["근거문구"]
                   for rid in src for v in ITEMS)
        check("등급·근거문구가 손실 없이 왕복", same)
        check("메타 보존", meta.get("판정모드") == "graded" and meta.get("문턱") == 2, str(meta))
        check("레코드수 자동 기록", meta.get("레코드수") == len(src), str(meta.get("레코드수")))
        dist = gradeio.distribution(got)
        check("등급 분포 합 = 칸 수", sum(dist.values()) == len(src) * len(ITEMS), str(dist))


# --------------------------------------------------------------------------- 6. 무호출
class _ExplodingRunner:
    """호출되면 즉시 실패하는 러너 — 스윕 경로가 LLM 을 안 쓰는지 증명한다."""
    name = "exploding"

    def generate(self, *a, **k):
        raise AssertionError("스윕 중 LLM 호출이 발생했다")

    def generate_mixed(self, *a, **k):
        raise AssertionError("스윕 중 LLM 호출이 발생했다")

    def count_tokens(self, *a, **k):
        raise AssertionError("스윕 중 토큰 계산이 발생했다")


def test_no_api(recs, tbl):
    print("\n[6] API 추가 호출 없음")
    pipe = Pipeline(_ExplodingRunner(), tbl)
    judged = {r.id: {v: {G: 2, "근거문구": None} for v in ITEMS} for r in recs[:5]}
    try:
        for r in recs[:5]:
            for th in range(GRADE_MIN_TH, GRADE_MAX_TH + 1):
                pipe.finalize(r, judged[r.id], threshold=th)
    except AssertionError as e:
        check("finalize 가 러너를 건드리지 않는다", False, str(e))
        return
    check("finalize 가 러너를 건드리지 않는다 (터지는 러너로 확인)", True)
    check("러너 None 으로도 Pipeline 을 만들어 스윕할 수 있다",
          Pipeline(None, tbl).finalize(recs[0], judged[recs[0].id], threshold=1) is not None)


# --------------------------------------------------------------------------- 1·7. 회귀·형식
def test_v24_regression(recs, tbl):
    print("\n[1] 기존 v24 회귀 (규칙 미배선 유지)")
    pipe = _pipe(recs, tbl)
    from pps import compare
    fired = sum(1 for r in recs if compare.check_v24_positive(r)[0])
    print(f"       check_v24_positive 가 참인 레코드 {fired}건 (배선 안 됨)")
    # 규칙이 배선되지 않았으므로 v24 는 LLM 등급만으로 결정된다
    off = 0
    for r in recs:
        judged = {"v24": {G: 0, "근거문구": None}}
        if pipe.finalize(r, judged, threshold=1)["v24"]["위반여부"] != 0:
            off += 1
    check("v24 는 등급 0 이면 항상 0 (규칙이 1 로 올리지 않는다)", off == 0, f"{off}건 위반")

    # 이진 모드 등급(0/3)은 문턱 1~3 어디서나 같은 판정을 낸다 → 회귀 안전
    diffs = 0
    for r in recs:
        judged = {v: {G: (prompts.GRADE_MAX if (i % 3 == 0) else 0), "근거문구": None}
                  for i, v in enumerate(ITEMS)}
        base = pipe.finalize(r, judged, threshold=1)
        for th in (2, 3):
            cur = pipe.finalize(r, judged, threshold=th)
            diffs += sum(1 for v in ITEMS
                         if base[v]["위반여부"] != cur[v]["위반여부"])
    check("이진 모드로 저장된 등급은 문턱 1~3 에서 동일한 판정 (회귀 안전)",
          diffs == 0, f"{diffs}칸 차이")


def test_submission_format(recs, tbl):
    print("\n[7] 제출 형식")
    pipe = _pipe(recs, tbl)
    judged = {r.id: {v: {G: 3, "근거문구": "이 공고는"} for v in ITEMS} for r in recs}
    rows = [submission.to_row(r.id, pipe.finalize(r, judged[r.id], threshold=2))
            for r in recs]
    check("행 수 = 레코드 수", len(rows) == len(recs))
    check("49열", all(set(row) == set(COLUMNS) for row in rows), str(len(COLUMNS)))
    vals = {row[v] for row in rows for v in ITEMS}
    check("v 열은 정수 0/1 만", vals <= {0, 1} and all(isinstance(x, int) for x in vals),
          str(vals))
    bad_abs = [v for row in rows for v in ITEMS if v in ABSENCE and row[f"e{v[1:]}"]]
    check("부재탐지 항목의 근거문구는 빈칸", not bad_abs, str(bad_abs[:3]))
    bad_zero = [v for row in rows for v in ITEMS if row[v] == 0 and row[f"e{v[1:]}"]]
    check("비위반 항목의 근거문구는 빈칸", not bad_zero, str(bad_zero[:3]))
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "s.csv")
        submission.write_csv(rows, p)
        errs = submission.validate(p, [r.id for r in recs])
        check("submission.validate 통과", not errs, str(errs[:3]))


def main() -> int:
    recs = load_records(os.path.join(OPEN, "dev.jsonl.gz"), limit=30)
    tbl = load_item_table(os.path.join(OPEN, "data"))
    print(f"레코드 {len(recs)}건 · 문턱 기본값 {GRADE_THRESHOLD_DEFAULT} "
          f"· 문턱 범위 {GRADE_MIN_TH}~{GRADE_MAX_TH}")

    test_v24_regression(recs, tbl)
    test_parse()
    test_threshold(recs, tbl)
    test_sweep_monotonic(recs, tbl)
    test_roundtrip(recs)
    test_no_api(recs, tbl)
    test_submission_format(recs, tbl)

    print("\n" + ("전체 PASS" if not _fails else f"FAIL {len(_fails)}건: {_fails}"))
    return 1 if _fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
