# -*- coding: utf-8 -*-
"""추론 파이프라인 — 게이팅 → 그룹별 LLM 호출 → 규칙 결합 → 제출 행 생성."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import evidence, gating, presence, prompts, pumnum
from .records import ABSENCE, ITEMS, Record
from .runner import GenConfig, run_with_fallback

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


# --------------------------------------------------------------------------- 파싱

def extract_json(text: str) -> Optional[Any]:
    text = (text or "").strip()
    if not text:
        return None
    for cand in (text, *(m.group(1) for m in _FENCE.finditer(text))):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            pass
    i, j = text.find("{"), text.rfind("}")
    if i >= 0 and j > i:
        try:
            return json.loads(text[i:j + 1])
        except json.JSONDecodeError:
            return None
    return None


def _as01(x: Any) -> int:
    if isinstance(x, bool):
        return int(x)
    if isinstance(x, (int, float)):
        return 1 if int(x) == 1 else 0
    if isinstance(x, str):
        return 1 if x.strip() in ("1", "위반", "true", "True", "Y", "y") else 0
    return 0


def parse_group(text: str, items: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """모델 출력 → {항목: {위반여부, 근거문구}}. 빠진 항목은 0/None 으로 채운다."""
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("판정"), dict):
        obj = obj["판정"]
    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for v in items:
        raw = obj.get(v) if isinstance(obj, dict) else None
        if not isinstance(raw, dict):
            missing.append(v)
            out[v] = {"위반여부": 0, "근거문구": None}
            continue
        ev = raw.get("근거문구", raw.get("evidence"))
        if ev is not None and not isinstance(ev, str):
            ev = str(ev)
        out[v] = {"위반여부": _as01(raw.get("위반여부", raw.get("violation", 0))),
                  "근거문구": ev}
    return out, missing


# --------------------------------------------------------------------------- 작업 단위

@dataclass
class Task:
    rec: Record
    group: prompts.Group
    items: List[str]
    messages: List[Dict[str, str]] = field(default_factory=list)
    ntok: int = 0
    # 대회 규칙 2-1) 각 공고마다 고정 LLM 정상 호출이 1회 이상 있어야 한다.
    # 없으면 '제출 요건 미충족'으로 무효 처리된다 — 시간 초과보다 나쁜 결과다.
    # 공고당 첫 작업에 이 표시를 달고, 워치독은 이 작업을 절대 건너뛰지 않는다.
    mandatory: bool = False


@dataclass
class Stats:
    n_records: int = 0
    n_calls: int = 0
    n_planned: int = 0
    n_skipped: int = 0
    n_empty: int = 0
    n_missing_items: int = 0
    n_evidence_kept: int = 0
    n_evidence_dropped: int = 0
    n_evidence_repaired: int = 0
    gated_cells: int = 0
    seconds: float = 0.0
    # 규칙 2-1) 공고당 고정 LLM 정상 호출 1회 이상. 0이 아니면 제출물이 무효 처리된다.
    records_without_call: int = 0

    def report(self) -> str:
        lines = [
            f"레코드 {self.n_records} · LLM 호출 {self.n_calls} "
            f"(공고당 {self.n_calls / max(1, self.n_records):.2f}회)",
            f"게이팅 차단 칸 {self.gated_cells} "
            f"({self.gated_cells / max(1, self.n_records * 24):.1%})",
            f"빈 출력 {self.n_empty} · 결손 항목 {self.n_missing_items}",
            f"근거문구 채택 {self.n_evidence_kept} "
            f"(그중 복원 {self.n_evidence_repaired}) · 폐기 {self.n_evidence_dropped}",
            f"소요 {self.seconds:.1f}s",
        ]
        if self.n_skipped:
            lines.insert(1, f"⏱ 시간예산으로 생략한 보강 호출 {self.n_skipped}"
                            f"/{self.n_planned} — 해당 항목은 0으로 제출됨")
        if self.records_without_call:
            lines.insert(0, f"❌ 규칙 위반: LLM 정상 응답이 0건인 공고 "
                            f"{self.records_without_call}건 "
                            f"— 제출 요건 미충족으로 무효 처리 대상")
        else:
            lines.insert(0, "✅ 규칙 2-1 충족: 모든 공고에 LLM 정상 응답 1건 이상")
        return "\n".join(lines)


# --------------------------------------------------------------------------- 파이프라인

class Pipeline:
    def __init__(
        self,
        runner,
        item_table: Dict[str, Dict[str, Any]],
        gosi: Optional[pumnum.Gosi] = None,
        law_texts: Optional[Dict[str, str]] = None,
        max_tokens: int = 1200,
        prompt_budget: int = 14848,
        seed: int = 20260826,
        use_rules: bool = True,
        deadline: Optional[float] = None,
    ):
        self.runner = runner
        self.tbl = item_table
        self.gosi = gosi
        self.law_texts = law_texts or {}
        self.max_tokens = max_tokens
        self.prompt_budget = prompt_budget
        self.seed = seed
        self.use_rules = use_rules
        # time.monotonic() 기준 마감 시각. 넘기면 남은 호출을 포기하고 0으로 낸다.
        # 2시간 초과는 '제출 오류'로 일일 제출 횟수가 차감되므로, 일부 항목을 0으로
        # 내더라도 파일을 남기는 쪽이 언제나 낫다.
        self.deadline = deadline
        self.stats = Stats()

    def _time_left(self) -> Optional[float]:
        return None if self.deadline is None else self.deadline - time.monotonic()

    # ---- 작업 생성 -------------------------------------------------------
    def plan(self, recs: Sequence[Record]) -> List[Task]:
        tasks: List[Task] = []
        for rec in recs:
            g = gating.gate(rec)
            self.stats.gated_cells += sum(1 for i in ITEMS if not g[i])
            made: List[Task] = []
            for group in prompts.GROUPS:
                items = [i for i in group.items if g[i]]
                if not items:
                    continue
                made.append(Task(rec, group, items))
            if not made:
                # 게이팅으로 전 항목이 차단돼도 반드시 한 번은 호출한다.
                g1 = prompts.GROUPS[0]
                made.append(Task(rec, g1, list(g1.items)))
            made[0].mandatory = True     # 규칙 2-1) 공고당 최소 1회 정상 호출
            tasks.extend(made)
        self.stats.n_records = len(recs)
        return tasks

    def render(self, task: Task) -> Task:
        budget = task.group.budget
        while True:
            msgs = prompts.build_messages(
                task.rec, task.group, task.items, self.tbl,
                gosi=self.gosi,
                law_text=self.law_texts.get(task.group.key, ""),
                budget=budget,
            )
            n = self.runner.count_tokens(msgs)
            if n <= self.prompt_budget - self.max_tokens or budget <= 1200:
                task.messages, task.ntok = msgs, n
                return task
            budget = int(budget * 0.8)

    # ---- 실행 -----------------------------------------------------------
    def run(self, recs: Sequence[Record], chunk: int = 64,
            progress: bool = True) -> Dict[str, Dict[str, Dict[str, Any]]]:
        t0 = time.time()
        tasks = [self.render(t) for t in self.plan(recs)]
        if progress:
            toks = sorted(t.ntok for t in tasks) or [0]
            print(f"  작업 {len(tasks)}개 · 프롬프트 토큰 중앙값 "
                  f"{toks[len(toks) // 2]:,} · 최대 {toks[-1]:,}")

        # 스키마가 같은 작업끼리 묶어야 제약 디코딩을 배치로 쓸 수 있다.
        # 다만 묶음 순서는 **그룹 우선순위**를 따른다 — 시간이 모자라 중간에 끊겨도
        # 특정 그룹만 통째로 날아가지 않고 모든 공고가 고르게 판정된다.
        order = {g.key: n for n, g in enumerate(prompts.GROUPS)}
        by_sig: Dict[Tuple[str, ...], List[int]] = {}
        for i, t in enumerate(tasks):
            by_sig.setdefault(tuple(t.items), []).append(i)
        sigs = sorted(by_sig, key=lambda s: order.get(prompts.GROUP_OF[s[0]].key, 99))

        outs: List[str] = [""] * len(tasks)
        self.stats.n_planned = len(tasks)
        done = 0

        def execute(indices: List[int], sig: Tuple[str, ...],
                    skippable: bool) -> bool:
            """한 스키마 묶음을 청크 단위로 실행. 중단했으면 False."""
            nonlocal done
            cfg = GenConfig(max_tokens=self.max_tokens, seed=self.seed,
                            schema=prompts.build_schema(sig))
            for s in range(0, len(indices), chunk):
                part = indices[s:s + chunk]
                if skippable:
                    left = self._time_left()
                    if left is not None:
                        per = (time.time() - t0) / max(1, done)
                        if left <= per * len(part) * 1.3:
                            print(f"  ⏱ 시간예산 소진 (남은 {left:.0f}s) → "
                                  f"보강 호출 중단")
                            return False
                res = run_with_fallback(
                    self.runner, [tasks[i].messages for i in part], cfg)
                for i, text in zip(part, res):
                    outs[i] = text
                done += len(part)
                if progress:
                    print(f"  {done}/{len(tasks)} … {time.time() - t0:.0f}s")
            return True

        # 1단계 — 필수 호출. 공고당 1회는 반드시 수행한다(규칙 2-1).
        #        시간이 모자라도 건너뛰지 않는다. 여기를 건너뛰면 '제출 요건 미충족'
        #        으로 제출물 전체가 무효 처리되며, 이는 시간 초과보다 나쁜 결과다.
        for sig in sigs:
            idxs = [i for i in by_sig[sig] if tasks[i].mandatory]
            if idxs:
                execute(idxs, sig, skippable=False)
        n_mandatory = sum(1 for t in tasks if t.mandatory)
        if progress:
            print(f"  ✓ 필수 호출 {n_mandatory}건 완료 ({time.time() - t0:.0f}s)")

        # 2단계 — 보강 호출. 남은 항목의 판정 품질을 올리지만, 없어도 제출은 유효하다.
        for sig in sigs:
            idxs = [i for i in by_sig[sig] if not tasks[i].mandatory]
            if idxs and not execute(idxs, sig, skippable=True):
                break

        self.stats.n_calls = done
        self.stats.n_skipped = len(tasks) - done

        # 규칙 준수 검증 — 공고당 정상 응답이 1건 이상인지
        ok_per_rec: Dict[str, int] = {r.id: 0 for r in recs}
        for t, text in zip(tasks, outs):
            if text:
                ok_per_rec[t.rec.id] += 1
        self.stats.records_without_call = sum(1 for n in ok_per_rec.values() if n == 0)

        judged: Dict[str, Dict[str, Dict[str, Any]]] = {r.id: {} for r in recs}
        for t, text in zip(tasks, outs):
            if not text:
                self.stats.n_empty += 1
            parsed, missing = parse_group(text, t.items)
            self.stats.n_missing_items += len(missing)
            judged[t.rec.id].update(parsed)

        self.stats.seconds = time.time() - t0
        return judged

    # ---- 결합 -----------------------------------------------------------
    def finalize(self, rec: Record,
                 judged: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """LLM 판정 + 게이팅 + 규칙을 결합해 최종 24항목을 만든다."""
        g = gating.gate(rec)
        src = rec.full_text
        out: Dict[str, Dict[str, Any]] = {}

        rule_hint = self._rule_hints(rec) if self.use_rules else {}

        for v in ITEMS:
            if not g[v]:
                out[v] = {"위반여부": 0, "근거문구": ""}
                continue
            cell = judged.get(v) or {"위반여부": 0, "근거문구": None}
            hit = 1 if cell.get("위반여부") == 1 else 0

            # 규칙이 확정적으로 아니라고 하면 내린다 (정밀도 우선)
            if rule_hint.get(v) == 0:
                hit = 0

            raw = cell.get("근거문구")
            ev = evidence.clean(raw, src, is_absence=(v in ABSENCE), is_violation=bool(hit))
            if hit and v not in ABSENCE:
                if ev:
                    self.stats.n_evidence_kept += 1
                    if raw and ev != raw.strip():
                        self.stats.n_evidence_repaired += 1
                elif raw:
                    self.stats.n_evidence_dropped += 1
            out[v] = {"위반여부": hit, "근거문구": ev}
        return out

    def _rule_hints(self, rec: Record) -> Dict[str, int]:
        """정규식이 확정적으로 '위반 아님'이라고 말할 수 있는 칸.

        부재탐지 항목은 '요구 문구가 문서에 있으면' 위반이 아니다.
        전체 원문을 절단 없이 보므로 LLM 보다 이 판단이 정확하다.
        """
        scans = presence.scan_record(rec)
        size = presence.size_restrictions(rec.full_text)
        hints: Dict[str, int] = {}
        if scans["v10"].present:
            hints["v10"] = 0                       # 직접생산확인 요구가 있다
        if size["중소기업"].present:
            hints["v11"] = 0                       # 중소기업자로 제한했다
            hints["v16"] = 0
        if size["소기업등"].present:
            hints["v18"] = 0                       # 소기업·소상공인으로 제한했다
        if scans["v20"].present:
            hints["v20"] = 0                       # 대기업 참여제한을 명시했다
        if not scans["_SW사업"].present:
            hints["v20"] = 0                       # SW사업이 아니다
        return hints
