# -*- coding: utf-8 -*-
"""추론 파이프라인 — 게이팅 → 그룹별 LLM 호출 → 규칙 결합 → 제출 행 생성."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import compare, evidence, gating, presence, prompts, pumnum, schedule
from .records import ABSENCE, ITEMS, Record
from .runner import GenConfig, run_with_fallback

_FENCE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.S)


# --------------------------------------------------------------------------- 파싱

def _repair_truncated(s: str) -> Optional[Any]:
    """토큰 예산에서 잘린 JSON 을 살린다.

    max_tokens 를 넘겨 중간에서 끊기면 통째로 버려져 그 그룹 전 항목이 0이 된다.
    마지막으로 완결된 항목까지만 남기고 닫아서 건질 수 있는 만큼 건진다.
    """
    s = s.strip()
    if not s.startswith("{"):
        i = s.find("{")
        if i < 0:
            return None
        s = s[i:]
    # 마지막으로 닫힌 중괄호 뒤에서 자르고 최상위를 닫는다
    for cut in range(len(s) - 1, 0, -1):
        if s[cut] != "}":
            continue
        cand = s[:cut + 1].rstrip().rstrip(",")
        for suffix in ("}", ""):
            try:
                return json.loads(cand + suffix)
            except json.JSONDecodeError:
                continue
    return None


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
            pass
    # 코드펜스 안이 잘린 경우까지 포함해 복구를 시도한다
    for cand in (*(m.group(1) for m in _FENCE.finditer(text)), text):
        got = _repair_truncated(cand)
        if got is not None:
            return got
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

        # 제약 디코딩이 없는 환경(로컬 검증 등)에서는 모델이 축약형을 낸다:
        #   {"v10": 0}  또는  {"v10": "0"}  대신  {"v10": {"위반여부": 0, "근거문구": null}}
        # 형식이 다르다고 버리면 그룹 전체가 0이 되므로 값만 받아 살린다.
        if isinstance(raw, (int, float, bool, str)):
            out[v] = {"위반여부": _as01(raw), "근거문구": None}
            continue

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
    n_rule_overrides: int = 0        # 규칙이 LLM 의 1 을 0 으로 내린 횟수
    n_verified: int = 0
    n_verify_dropped: int = 0

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
        if self.n_verified:
            lines.append(f"2단계 검증 {self.n_verified}건 중 {self.n_verify_dropped}건 취소")
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

        if getattr(self.runner, "per_request_schema", False):
            # 개발용(API·Mock) 경로 — 요청마다 스키마를 따로 보낼 수 있으므로
            # 스키마별로 쪼개지 않고 전체를 한 풀에서 병렬 실행한다.
            # (스키마별 순차 실행은 워커가 놀아 dev 20건에 10분이 걸렸다.)
            # 필수 호출을 앞에 두어 시간이 끊겨도 규칙 2-1 이 먼저 충족되게 한다.
            ordered = sorted(range(len(tasks)),
                             key=lambda i: (not tasks[i].mandatory,
                                            order.get(tasks[i].group.key, 99)))
            cfgs = [GenConfig(max_tokens=self.max_tokens, seed=self.seed,
                              schema=prompts.build_schema(tuple(tasks[i].items)))
                    for i in ordered]

            def _tick(_):
                nonlocal done
                done += 1
                if progress and done % 20 == 0:
                    print(f"  {done}/{len(tasks)} … {time.time() - t0:.0f}s")

            res = self.runner.generate_mixed(
                [tasks[i].messages for i in ordered], cfgs, on_done=_tick)
            for i, text in zip(ordered, res):
                outs[i] = text
            done = len(tasks)
        else:
            # 제출용(vLLM) 경로 — 같은 스키마끼리 묶어야 제약 디코딩 배치가 효율적이다.
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

    # ---- 2단계 검증 -------------------------------------------------------
    def verify(self, recs: Sequence[Record],
               judged: Dict[str, Dict[str, Dict[str, Any]]],
               chunk: int = 64, progress: bool = True) -> Dict[str, set]:
        """1로 판정된 칸만 다시 물어 과잉 판정을 걷어낸다.

        예측률이 실제 위반율보다 높으면 정밀도에 천장이 생긴다.
        dev200b 실측: 평균 예측률 4.29% vs 추정 실제 양성률 2% → Macro F1 상한 0.4883.
        (실제 리더보드 0.42458 과 거의 일치했다.)
        정확도를 올리는 것만으로는 이 천장을 못 넘고 **예측을 줄여야** 한다.

        반환: {공고id: 취소된 항목 집합}
        """
        by_rec = {r.id: r for r in recs}
        targets: List[Tuple[str, str, Optional[str]]] = []
        for rid, cells in judged.items():
            if rid not in by_rec:
                continue
            g = gating.gate(by_rec[rid])
            for item, cell in cells.items():
                if cell.get("위반여부") == 1 and g.get(item, True):
                    targets.append((rid, item, cell.get("근거문구")))

        if not targets:
            return {}
        if progress:
            print(f"  [2단계 검증] 대상 {len(targets)}건")

        msgs = [prompts.build_verify_messages(by_rec[rid], item, ev, self.tbl)
                for rid, item, ev in targets]
        cfg = GenConfig(max_tokens=200, seed=self.seed, schema=prompts.VERIFY_SCHEMA)

        t0 = time.time()
        if getattr(self.runner, "per_request_schema", False):
            outs = self.runner.generate_mixed(msgs, [cfg] * len(msgs))
        else:
            outs = []
            for s in range(0, len(msgs), chunk):
                outs.extend(run_with_fallback(self.runner, msgs[s:s + chunk], cfg))

        dropped: Dict[str, set] = {}
        kept = 0
        for (rid, item, _), text in zip(targets, outs):
            obj = extract_json(text)
            # 판단 불가(빈 출력·파싱 실패)면 원래 판정을 유지한다 —
            # 검증 실패를 이유로 재현율을 잃지 않는다.
            if not isinstance(obj, dict):
                kept += 1
                continue
            if _as01(obj.get("위반유지", obj.get("유지", 1))) == 1:
                kept += 1
            else:
                dropped.setdefault(rid, set()).add(item)

        n_drop = sum(len(s) for s in dropped.values())
        self.stats.n_verified = len(targets)
        self.stats.n_verify_dropped = n_drop
        if progress:
            print(f"  [2단계 검증] 유지 {kept} · 취소 {n_drop} "
                  f"({n_drop / max(1, len(targets)):.0%}) · {time.time() - t0:.0f}s")
        return dropped

    # ---- 결합 -----------------------------------------------------------
    def finalize(self, rec: Record,
                 judged: Dict[str, Dict[str, Any]],
                 dropped: Optional[set] = None) -> Dict[str, Dict[str, Any]]:
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
            if dropped and v in dropped:      # 2단계 검증에서 취소된 칸
                hit = 0

            # 규칙이 확정적으로 판단한 칸은 규칙을 따른다
            forced = rule_hint.get(v)
            if forced is not None and forced != hit:
                hit = forced
                self.stats.n_rule_overrides += 1

            raw = cell.get("근거문구")
            # v24: 인용한 금액이 등록값과 일치하면 '상이'가 아니다.
            # dev 측정에서 v24 거짓양성 9/20 이 전부 이 유형이었다.
            if v == "v24" and hit and compare.amount_is_matching_quote(rec, raw):
                hit = 0
                self.stats.n_rule_overrides += 1
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
        """규칙이 확정적으로 말할 수 있는 칸. 값 0=위반 아님, 1=위반.

        1 로 올리는 것은 규칙이 LLM 보다 확실히 나은 항목에만 쓴다.
        지금은 v23 뿐이다 — dev 41건(협상+지방)에서 규칙 F1 0.909 vs LLM 0.000.
        """
        scans = presence.scan_record(rec)
        hints: Dict[str, int] = {}

        # ⚠️ v10·v11 의 "문구가 있으면 0" 강제는 제거했다 (2026-09-07 감사).
        #    dev 200건에서 v10 진짜 양성 7건 중 4건, v11 6건 중 2건을 이 규칙이 죽였다.
        #    '직접생산'은 계약조건 상투문구("계약상대자가 직접생산 확인기준을 위반한
        #    사실을 확인한 경우…")와 법령 인용에 늘 등장해 133/200건에 걸린다.
        #    문구의 존재 ≠ 참가자격 요구. 존재 여부는 프롬프트 입력(full_doc)으로
        #    LLM 이 이미 보고 있으므로 규칙으로 덮어쓸 이유가 없다.
        #    부재탐지 = 정규식이 낫다는 원칙은 "문구가 정형적일 때"만 성립한다(v20 처럼).

        if scans["v20"].present:
            hints["v20"] = 0                       # 대기업 참여제한을 명시했다
        if not scans["_SW사업"].present:
            hints["v20"] = 0                       # SW사업이 아니다

        # v23 — 설명회일과 제안서 제출마감일의 간격을 조문 기준과 대조한다.
        # 판단이 서는 경우(True/False)만 반영하고, 보류(None)는 LLM 에 맡긴다.
        v23, _ = schedule.check_v23(rec)
        if v23 is not None:
            hints["v23"] = 1 if v23 else 0
        return hints
