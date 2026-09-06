# -*- coding: utf-8 -*-
"""오프라인 채점 — 리더보드 산식(Macro F1) 재현.

산식: 24개 항목 각각에 대해 위반(=1) 클래스의 F1을 계산하고 단순평균.
평가 데이터는 24항목 모두에 양성이 있으므로 F1 미정의는 발생하지 않는다.
다만 dev 200건 같은 부분집합에서는 양성이 0인 항목이 생길 수 있어,
그 경우를 명시적으로 구분한다(무시하지 않고 드러낸다 — 조용히 평균에서 빼면
개선을 착각하게 된다).
"""
from __future__ import annotations

import csv
import io
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

from .records import ITEMS


@dataclass
class ItemScore:
    item: str
    tp: int
    fp: int
    fn: int
    tn: int

    @property
    def support(self) -> int:
        """정답 양성 개수."""
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        return self.tp + self.fp

    @property
    def precision(self) -> float:
        return self.tp / self.predicted if self.predicted else 0.0

    @property
    def recall(self) -> float:
        return self.tp / self.support if self.support else 0.0

    @property
    def f1(self) -> Optional[float]:
        """양성 클래스 F1. 정답 양성이 0이면 None(미정의)."""
        if self.support == 0:
            return None
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0


@dataclass
class Report:
    items: Dict[str, ItemScore]
    n_rows: int

    @property
    def scored_items(self) -> List[ItemScore]:
        return [s for s in self.items.values() if s.f1 is not None]

    @property
    def undefined_items(self) -> List[str]:
        return [k for k, s in self.items.items() if s.f1 is None]

    @property
    def macro_f1(self) -> float:
        """정의된 항목만 평균. 리더보드는 24항목 전부 정의되므로 동일해진다."""
        xs = [s.f1 for s in self.scored_items]
        return sum(xs) / len(xs) if xs else 0.0

    @property
    def macro_f1_strict(self) -> float:
        """미정의 항목을 0으로 보는 하한. dev에서 항목 누락을 숨기지 않기 위한 지표."""
        xs = [(s.f1 or 0.0) for s in self.items.values()]
        return sum(xs) / len(xs) if xs else 0.0

    def table(self) -> str:
        w = []
        w.append(f"{'항목':>5} {'정답+':>6} {'예측+':>6} {'TP':>4} {'FP':>4} {'FN':>4} "
                 f"{'정밀도':>7} {'재현율':>7} {'F1':>7}")
        w.append("-" * 68)
        for k in ITEMS:
            s = self.items[k]
            f1 = "  —  " if s.f1 is None else f"{s.f1:.4f}"
            w.append(f"{k:>5} {s.support:>6} {s.predicted:>6} {s.tp:>4} {s.fp:>4} {s.fn:>4} "
                     f"{s.precision:>7.4f} {s.recall:>7.4f} {f1:>7}")
        w.append("-" * 68)
        w.append(f"Macro F1 = {self.macro_f1:.4f}   (정의된 항목 {len(self.scored_items)}/24)")
        if self.undefined_items:
            w.append(f"※ 정답 양성 0인 항목: {', '.join(self.undefined_items)} "
                     f"→ strict {self.macro_f1_strict:.4f}")
        return "\n".join(w)

    def weakest(self, n: int = 5) -> List[ItemScore]:
        """F1이 낮은 순. 개선 우선순위."""
        xs = [s for s in self.items.values() if s.f1 is not None]
        return sorted(xs, key=lambda s: s.f1)[:n]


def score(
    pred: Dict[str, Dict[str, int]],
    gold: Dict[str, Dict[str, int]],
) -> Report:
    """pred/gold: {공고id: {'v1': 0/1, ...}}

    gold에 있는 id만 채점한다. pred에 없는 id는 전항목 0으로 간주한다.
    """
    scores: Dict[str, ItemScore] = {}
    for item in ITEMS:
        tp = fp = fn = tn = 0
        for rid, g in gold.items():
            gy = int(g.get(item, 0))
            py = int(pred.get(rid, {}).get(item, 0))
            if gy == 1 and py == 1:
                tp += 1
            elif gy == 0 and py == 1:
                fp += 1
            elif gy == 1 and py == 0:
                fn += 1
            else:
                tn += 1
        scores[item] = ItemScore(item, tp, fp, fn, tn)
    return Report(items=scores, n_rows=len(gold))


# --------------------------------------------------------------------------- 라벨 I/O

def load_labels(path: str) -> Dict[str, Dict[str, int]]:
    """dev_labels.csv → {id: {v1..v24: 0/1}}"""
    out: Dict[str, Dict[str, int]] = {}
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {k: int(row[k] or 0) for k in ITEMS}
    return out


def load_evidence(path: str) -> Dict[str, Dict[str, str]]:
    """dev_labels.csv → {id: {v1..v24: 근거문구}} (e열을 v키로 맞춰 반환)"""
    out: Dict[str, Dict[str, str]] = {}
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {f"v{i}": (row.get(f"e{i}") or "") for i in range(1, 25)}
    return out


def load_submission(path: str) -> Dict[str, Dict[str, int]]:
    """submission.csv → {id: {v1..v24: 0/1}}"""
    out: Dict[str, Dict[str, int]] = {}
    with io.open(path, encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            out[row["id"]] = {k: int(row[k] or 0) for k in ITEMS}
    return out


# --------------------------------------------------------------------------- 게이팅 진단

@dataclass
class GateAudit:
    """게이팅이 진짜 양성을 죽였는지 감사.

    게이팅으로 강제 0이 된 칸 중 정답이 1인 것 = 회복 불가능한 손실.
    이 값이 0이 아니면 게이팅 규칙이 틀린 것이다.
    """
    item: str
    killed_positive: int      # 게이팅이 죽인 진짜 양성 (치명적)
    killed_negative: int      # 게이팅이 막아준 거짓양성 기회 (이득)
    gated_total: int          # 게이팅으로 0 강제된 전체 칸 수


def audit_gates(
    gates: Dict[str, Dict[str, bool]],
    gold: Dict[str, Dict[str, int]],
) -> List[GateAudit]:
    """gates: {id: {v1: applicable?}} — False면 강제 0."""
    out = []
    for item in ITEMS:
        kp = kn = tot = 0
        for rid, g in gold.items():
            if gates.get(rid, {}).get(item, True):
                continue
            tot += 1
            if int(g.get(item, 0)) == 1:
                kp += 1
            else:
                kn += 1
        out.append(GateAudit(item, kp, kn, tot))
    return out


def format_gate_audit(audits: Sequence[GateAudit], n_rows: int) -> str:
    w = [f"{'항목':>5} {'차단':>6} {'차단율':>7} {'죽인양성':>9} {'막은음성':>9}"]
    w.append("-" * 42)
    fatal = 0
    for a in audits:
        if a.gated_total == 0:
            continue
        mark = "  ← 치명" if a.killed_positive else ""
        fatal += a.killed_positive
        w.append(f"{a.item:>5} {a.gated_total:>6} {a.gated_total / n_rows:>6.1%} "
                 f"{a.killed_positive:>9} {a.killed_negative:>9}{mark}")
    w.append("-" * 42)
    w.append(f"게이팅이 죽인 진짜 양성 총 {fatal}건 "
             + ("✅ 안전" if fatal == 0 else "❌ 규칙 재검토 필요"))
    return "\n".join(w)
