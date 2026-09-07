#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""v21(공동수급 최소지분율) 원문 조사 — 규칙 설계용. LLM 호출 없음.

dev 라벨 기준으로 양성/음성 공고에서 '지분율' 관련 문장을 뽑아 보여준다.
정답 근거문구(e21)가 6건 중 5건 비어 있어, 위반의 실제 형태를 원문에서 확인해야 한다.

사용:
    python3 tools/probe_v21.py            # 양성 전부 + 음성 표본
    python3 tools/probe_v21.py --all      # 전체 200건 지분율 표기 통계
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pps.records import load_records  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPEN = os.path.join(ROOT, "open")

# 지분율 표기: "최소 지분율 3% 이상", "지분율은 100분의 5 이상", "출자비율 5%"
PCT = re.compile(
    r"(지분(?:율|참여\s*비율)?|출자\s*비율)[^\n%]{0,40}?"
    r"(?:(\d{1,3}(?:\.\d+)?)\s*%|100\s*분의\s*(\d{1,3}))")
ANY = re.compile(r"지분|출자\s*비율")
공동 = re.compile(r"공동\s*(?:수급|계약|도급|이행)|분담\s*이행")


def shares(text: str):
    """본문에서 (표기문자열, 백분율) 목록."""
    out = []
    for m in PCT.finditer(text):
        pct = m.group(2) or m.group(3)
        if pct is None:
            continue
        seg = text[max(0, m.start() - 60): m.end() + 40].replace("\n", " ").strip()
        out.append((seg, float(pct)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()

    recs = {r.id: r for r in load_records(os.path.join(OPEN, "dev.jsonl.gz"))}
    rows = list(csv.DictReader(open(os.path.join(OPEN, "dev_labels.csv"),
                                    encoding="utf-8")))
    gold = {r["id"]: r for r in rows}
    pos = [r["id"] for r in rows if r.get("v21") == "1"]

    if args.all:
        n공동 = n지분 = n수치 = 0
        for rid, rec in recs.items():
            t = rec.full_text
            has공동 = bool(공동.search(t))
            has지분 = bool(ANY.search(t))
            sh = shares(t)
            n공동 += has공동
            n지분 += has지분
            n수치 += bool(sh)
        n = len(recs)
        print(f"dev {n}건 — 공동수급 언급 {n공동} ({n공동/n:.1%}) · "
              f"지분 언급 {n지분} ({n지분/n:.1%}) · 지분율 수치 추출 {n수치} ({n수치/n:.1%})")
        print(f"v21 양성 {len(pos)}건")
        for rid in pos:
            sh = shares(recs[rid].full_text)
            vals = sorted({v for _, v in sh})
            print(f"  {rid:>14} {recs[rid].적용계약법:>6} "
                  f"공동언급={'O' if 공동.search(recs[rid].full_text) else 'X'} "
                  f"추출값={vals or '-'}")
        return 0

    for rid in pos:
        rec = recs.get(rid)
        if rec is None:
            print(f"{rid} 레코드 없음")
            continue
        t = rec.full_text
        ev = (gold[rid].get("e21") or "").strip()
        print(f"\n===== {rid} | {rec.적용계약법} {rec.업무구분} "
              f"{rec.추정가격:,}원 | 정답근거={'있음' if ev else '빈칸'}")
        if ev:
            print(f"  [정답근거] {ev[:200]}")
        sh = shares(t)
        if sh:
            for seg, v in sh[:4]:
                print(f"  [수치 {v}%] …{seg[:170]}")
        else:
            seen = set()
            for m in ANY.finditer(t):
                seg = t[max(0, m.start() - 70): m.end() + 110]
                seg = " ".join(seg.split())
                if seg[:40] in seen:
                    continue
                seen.add(seg[:40])
                print(f"  [수치없음] …{seg[:175]}")
                if len(seen) >= 3:
                    break
            if not seen:
                print("  (본문에 '지분/출자비율' 언급 자체가 없다)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
