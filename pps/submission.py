# -*- coding: utf-8 -*-
"""제출 CSV 작성·자가검증.

규약 (README §5 · 평가 탭)
  · 49열: id, v1..v24, e1..e24
  · 행 수 = 입력 건수, id 유일·전건 존재
  · v 는 정수 0/1 만 (확률·True/False·"1" 불가)
  · e 는 원문 부분문자열 500자 이하, 비위반·부재탐지는 빈칸
  · UTF-8 BOM 없음, NFC, RFC4180 quoting, 수식 접두(= + @) 금지
"""
from __future__ import annotations

import csv
import io
import os
import unicodedata
from typing import Any, Dict, List, Sequence

from .records import ABSENCE, COLUMNS, EVIDENCE_MAX, ITEMS


def to_row(rec_id: str, judgment: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    row: Dict[str, Any] = {"id": rec_id}
    for i, v in enumerate(ITEMS, 1):
        cell = judgment.get(v) or {"위반여부": 0, "근거문구": ""}
        row[v] = 1 if cell.get("위반여부") == 1 else 0
        row[f"e{i}"] = cell.get("근거문구") or ""
    return row


def empty_row(rec_id: str) -> Dict[str, Any]:
    return to_row(rec_id, {v: {"위반여부": 0, "근거문구": ""} for v in ITEMS})


def write_csv(rows: Sequence[Dict[str, Any]], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({k: unicodedata.normalize("NFC", str(r[k])) for k in COLUMNS})


def validate(path: str, expected_ids: Sequence[str]) -> List[str]:
    """제출 전 자가검증. 빈 리스트면 통과."""
    errs: List[str] = []
    with io.open(path, "r", encoding="utf-8", newline="") as f:
        rd = csv.reader(f)
        header = next(rd, None)
        rows = list(rd)

    if header != COLUMNS:
        return [f"헤더 불일치: {len(header or [])}열 (기대 {len(COLUMNS)})"]
    if len(rows) != len(expected_ids):
        errs.append(f"행 수 {len(rows)} ≠ 입력 {len(expected_ids)}")

    ids = [r[0] for r in rows]
    if len(set(ids)) != len(ids):
        errs.append("id 중복")
    missing = set(expected_ids) - set(ids)
    if missing:
        errs.append(f"id 누락 {len(missing)}건 (예: {sorted(missing)[:3]})")

    absence_idx = {COLUMNS.index("e" + v[1:]) for v in ABSENCE}
    for r in rows:
        if len(r) != len(COLUMNS):
            errs.append(f"{r[0]}: 열 수 {len(r)}")
            continue
        if any(x not in ("0", "1") for x in r[1:25]):
            errs.append(f"{r[0]}: 위반여부에 0/1 아닌 값")
        if any(len(x) > EVIDENCE_MAX for x in r[25:]):
            errs.append(f"{r[0]}: 근거문구 {EVIDENCE_MAX}자 초과")
        if any(r[j] for j in absence_idx):
            errs.append(f"{r[0]}: 부재탐지 항목에 근거문구")
        if any(x.startswith(("=", "+", "@")) for x in r[25:]):
            errs.append(f"{r[0]}: 수식 접두 근거문구")
        if len(errs) > 20:
            errs.append("… 이하 생략")
            break
    return errs


def verify_evidence_in_source(
    rows: Sequence[Dict[str, Any]],
    sources: Dict[str, str],
) -> List[str]:
    """근거문구가 실제로 해당 공고 원문의 부분문자열인지 확인.

    2차 평가에서 부분문자열이 아닌 근거는 무효 처리된다.
    """
    bad: List[str] = []
    for r in rows:
        src = unicodedata.normalize("NFC", sources.get(r["id"], ""))
        for i in range(1, 25):
            e = r.get(f"e{i}") or ""
            if e and e not in src:
                bad.append(f"{r['id']} e{i}: 원문 불일치 {e[:40]!r}")
    return bad
