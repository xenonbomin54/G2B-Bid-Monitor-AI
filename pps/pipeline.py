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

# --------------------------------------------------------------------------- 판정 문턱
# **문턱은 이 한 값으로만 관리한다.** Pipeline 이 기본값으로 쓰고,
# tools/evaluate.py · build_submit.py 는 넘기지 않으면 이 값을 그대로 받는다.
#
# 위반등급(0~3) 이 이 값 이상이면 최종 1 이다.
# 1 이면 '의심'까지 위반으로 본다(재현율 우선), 3 이면 '명백'만 본다(정밀도 우선).
# 이진 모드에서는 1 → 3, 0 → 0 으로 담기므로 1~3 어디서나 기존 판정이 재현된다.
#
# ⚠️ 실측 없이 기본값을 바꾸지 말 것. `python3 tools/sweep_threshold.py` 로
#    저장된 등급을 재채점해 항목별 P/R/F1 을 보고 정한다.
# 문턱 3 — 저장된 등급으로 스윕해 정했다(API 0회). 세 독립 실행 전부에서 3 이 이겼다.
#   dev200n 0.6822 → 0.6967 (+0.0145)   FP 123→106 · TP 127→125
#   dev200m 0.6788 → 0.6923 (+0.0135)
#   dev200o 0.6708 → 0.6834 (+0.0126)
# 이유: 등급 2 는 19칸(0.5%)뿐인데 그중 17개가 FP 다 — **89% 오탐**.
# 모델이 '확실하지 않다'고 표시한 칸은 실제로 대부분 틀린다. 미약하지만 실재하는 신호다.
# 실제 분포는 dev 보다 정밀도가 낮으므로(0.331 vs 0.508) LB 에서 이 거래가 더 유리하다.
# 문턱 1 과 2 는 결과가 완전히 같다(등급 1 을 쓴 칸이 0개).
GRADE_THRESHOLD_DEFAULT = 3

# 문턱이 가질 수 있는 값의 범위. 0 은 모든 칸을 위반으로 만들어 의미가 없다.
GRADE_MIN_TH = 1
GRADE_MAX_TH = prompts.GRADE_MAX


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


def _as_grade(x: Any, binary: bool) -> int:
    """모델이 낸 값을 위반등급 0~3 으로 읽는다.

    binary=True (위반여부 0/1 로 받은 값)
      1 → GRADE_MAX, 0 → 0 으로 **끝값에 붙인다.**
      ⚠️ 여기서 1 을 등급 1 로 담으면 기본 문턱 2 에서 전부 0 이 되어
         이진 모드가 통째로 망가진다. 어떤 문턱(1~3)에서도 원래 판정이
         그대로 재현되어야 한다 — 회귀 안전의 핵심이다.

    binary=False (위반등급 0~3 으로 받은 값)
      범위를 벗어나면 0~3 으로 잘라 맞춘다.
    """
    if binary:
        return prompts.GRADE_MAX if _as01(x) else 0
    if isinstance(x, bool):
        return prompts.GRADE_MAX if x else 0
    if isinstance(x, (int, float)):
        return max(prompts.GRADE_MIN, min(prompts.GRADE_MAX, int(x)))
    if isinstance(x, str):
        s = x.strip()
        if s.isdigit():
            return max(prompts.GRADE_MIN, min(prompts.GRADE_MAX, int(s)))
        return prompts.GRADE_MAX if _as01(s) else 0
    return 0


def parse_select(text: str, items: Sequence[str]) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """선택형 출력 파싱 — `위반항목` 배열에 담긴 것만 위반이다.

    §prompts 선택형 출력 참조. 고르지 않은 항목은 0 이다 — 이것이 이 구조의 요점이다.
    "위반 없음"이 24번의 부정이 아니라 **빈 배열 한 번**으로 표현된다.

    배열을 못 읽으면 `missing` 에 전 항목을 담아 호출 실패로 집계한다 —
    빈 배열(정상적인 '위반 없음')과 파싱 실패를 구분해야 한다.
    """
    obj = extract_json(text)
    out: Dict[str, Dict[str, Any]] = {
        v: {prompts.GRADE_KEY: 0, "근거문구": None} for v in items}
    if not isinstance(obj, dict):
        return out, list(items)
    arr = obj.get(prompts.SELECT_KEY)
    if arr is None:
        # 키 이름이 다르게 나온 경우 — 배열 값을 하나 찾아본다
        arr = next((v for v in obj.values() if isinstance(v, list)), None)
    if not isinstance(arr, list):
        return out, list(items)
    want = set(items)
    for e in arr:
        if isinstance(e, str):
            if e in want:
                out[e] = {prompts.GRADE_KEY: prompts.GRADE_MAX, "근거문구": None}
            continue
        if not isinstance(e, dict):
            continue
        it = e.get("항목") or e.get("item")
        if it not in want:
            continue
        ev = e.get("근거문구", e.get("evidence"))
        if ev is not None and not isinstance(ev, str):
            ev = str(ev)
        out[it] = {prompts.GRADE_KEY: prompts.GRADE_MAX, "근거문구": ev}
    return out, []


def parse_group(text: str, items: Sequence[str],
                graded: bool = False) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """모델 출력 → {항목: {위반등급, 근거문구}}. 빠진 항목은 0/None 으로 채운다.

    **여기서는 최종 0/1 을 만들지 않는다.** LLM 원본 등급을 그대로 보존하고,
    문턱 적용은 `Pipeline.finalize` 가 한다 — 그래야 한 번 호출한 결과로
    문턱만 바꿔 가며 재채점할 수 있다(tools/sweep_threshold.py).

    이진 모드(graded=False)에서는 1 → GRADE_MAX, 0 → 0 으로 올려 담는다.
    따라서 문턱 1~3 어디서도 기존 판정이 그대로 재현된다(회귀 안전).
    """
    obj = extract_json(text)
    if isinstance(obj, dict) and isinstance(obj.get("판정"), dict):
        obj = obj["판정"]
    key = prompts.GRADE_KEY if graded else prompts.BINARY_KEY
    out: Dict[str, Dict[str, Any]] = {}
    missing: List[str] = []
    for v in items:
        raw = obj.get(v) if isinstance(obj, dict) else None

        # 제약 디코딩이 없는 환경(로컬 검증 등)에서는 모델이 축약형을 낸다:
        #   {"v10": 0}  또는  {"v10": "0"}  대신  {"v10": {...}}
        # 형식이 다르다고 버리면 그룹 전체가 0이 되므로 값만 받아 살린다.
        if isinstance(raw, (int, float, bool, str)):
            out[v] = {prompts.GRADE_KEY: _as_grade(raw, binary=not graded),
                      "근거문구": None}
            continue

        if not isinstance(raw, dict):
            missing.append(v)
            out[v] = {prompts.GRADE_KEY: 0, "근거문구": None}
            continue
        ev = raw.get("근거문구", raw.get("evidence"))
        if ev is not None and not isinstance(ev, str):
            ev = str(ev)
        # 요청한 키가 없으면 다른 키도 받아 본다 — 모델이 모드를 헷갈릴 수 있다.
        # **값을 어느 키에서 얻었는지가 해석을 정한다** — 위반여부에서 온 1 은
        # 등급 1 이 아니라 GRADE_MAX 다.
        if key in raw:
            val, binary = raw[key], (key == prompts.BINARY_KEY)
        else:
            other = prompts.BINARY_KEY if graded else prompts.GRADE_KEY
            if other in raw:
                val, binary = raw[other], (other == prompts.BINARY_KEY)
            else:
                val, binary = raw.get("violation", 0), True
        out[v] = {prompts.GRADE_KEY: _as_grade(val, binary=binary), "근거문구": ev}
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
    # Architecture A — 1단계 사실추출에 성공한 공고 수
    n_extracted: int = 0
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
        # 1200 으로 충분하다 — 늘렸다가 실측으로 되돌렸다(2026-09-07).
        #
        # dev200g 의 '결손 항목 392개(8.2%)' 를 토큰 상한 절단으로 의심해 2000 으로
        # 올렸는데, `.cache/api` 1,200건을 열어 보니 원인이 달랐다.
        #   정상 응답 1,081건 — 길이 중앙 149자 · 90% 318자 · **최대 580자**
        #   절단   119건(9.9%) — **전부 200자 미만, 대부분 1~9자** (예: '{\\n  "v12":')
        # 정상 응답이 최대 580자니 1200 토큰 상한에 닿은 적이 없다. 절단은 조각난
        # 1~9자짜리 — NVIDIA 엔드포인트의 **연결 끊김**이다(토큰 상한이 아니다).
        # 제출 경로는 로컬 VllmRunner 라 네트워크가 없다 → 이 문제 자체가 없다.
        # 즉 결손 392칸은 **dev 하네스의 계측 오차**이고 제출본의 재현율 누출이 아니다.
        #
        # 함의: dev 측정값은 그룹 호출의 약 10%를 잃은 상태였다 → 우리 dev 점수는
        #       실제보다 **낮게** 나왔다. 놓침의 '그룹 전체 침묵'도 일부는 이것이다.
        #       고칠 곳은 max_tokens 가 아니라 개발용 러너의 재시도다(runner.py 참조).
        max_tokens: int = 1200,
        prompt_budget: int = 14848,
        seed: int = 20260826,
        use_rules: bool = True,
        deadline: Optional[float] = None,
        graded: bool = False,
        grade_threshold: int = GRADE_THRESHOLD_DEFAULT,
        select: bool = False,
        dual: bool = False,
        extract: bool = False,
        lang: str = "ko",
    ):
        self.runner = runner
        self.tbl = item_table
        self.gosi = gosi
        self.law_texts = law_texts or {}
        self.max_tokens = max_tokens
        # 양면 판단은 항목당 `적법근거`(≤80자) 가 더 붙는다.
        # 축약 전(300자) 실측 최대가 931자였고 축약판은 그보다 짧다.
        # 1200 으로도 대개 들어가지만 9항목 그룹의 상한을 감당하려면 여유가 필요하다.
        if dual and self.max_tokens < 1800:
            self.max_tokens = 1800
        self.prompt_budget = prompt_budget
        self.seed = seed
        self.use_rules = use_rules
        # graded: LLM 에 위반등급(0~3)을 요구한다. False 면 기존 이진 판정.
        # grade_threshold: 등급 ≥ 이 값이면 최종 1. 저장된 등급을 재채점할 때
        #   이 값만 바꾸면 되므로 API 재호출이 필요 없다.
        self.graded = graded
        self.grade_threshold = grade_threshold
        # select: 항목별 판정 대신 **위반 항목만 고르는** 배열을 받는다.
        #   dev200n 실측 — 예측 248 vs 정답 153(1.62배 과예측), 정답0 공고 112건에서 FP 62건.
        #   F1 = 2TP/(예측+정답) 이므로 과예측을 줄이는 것이 가장 큰 레버다.
        self.select = select
        # extract: 판정 전에 공고당 1회 **사실 추출**을 돌리고, 판정 단계에는
        # 원문 대신 그 사실표를 준다(§prompts 사실 추출, Architecture A).
        # dev200 FP 105건 중 59건(56%)이 '문구는 정확히 찾았으나 적용 판단 실패'였고,
        # 원문을 직접 보며 판정하는 구조가 그 경로를 열어 준다는 것이 분석의 결론이다.
        self.extract = extract
        # lang="en": 지시문만 영어로 바꾼 A/B 실험판(§prompts_en).
        # 공고 원문·사실 블록·스키마 키·등급 체계는 한국어판과 동일하다.
        self.lang = lang
        self.facts: Dict[str, Any] = {}
        # dual: 등급 앞에 `적법근거` 를 두어 반증을 먼저 탐색시킨다.
        self.dual = dual
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

    def _extract(self, recs: Sequence[Record], chunk: int = 64,
                 progress: bool = True) -> Dict[str, Any]:
        """1단계 — 공고당 1회, 원문에서 사실만 뽑는다.

        실패하면 그 공고만 facts 가 없어 원문 판정으로 되돌아간다(§render).
        전체가 무너지지 않게 하는 것이 중요하다 — 추출은 보조 단계이지 관문이 아니다.
        """
        t0 = time.time()
        schema = prompts.build_extract_schema()
        msgs = []
        for r in recs:
            b = self.prompt_budget - self.max_tokens
            m = prompts.build_extract_messages(r, budget=None)
            if self.runner.count_tokens(m) > b:
                # 길면 본문을 줄여 다시 만든다 — 자르는 지점은 문서 앞쪽을 살린다.
                m = prompts.build_extract_messages(r, budget=max(4000, b * 2))
            msgs.append(m)
        cfg = GenConfig(max_tokens=1600, seed=self.seed, schema=schema)
        outs: List[str] = [""] * len(msgs)
        for s in range(0, len(msgs), chunk):
            part = list(range(s, min(s + chunk, len(msgs))))
            res = run_with_fallback(self.runner, [msgs[i] for i in part], cfg)
            for i, text in zip(part, res):
                outs[i] = text
            if progress:
                print(f"  [사실추출] {min(s + chunk, len(msgs))}/{len(msgs)} … "
                      f"{time.time() - t0:.0f}s")
        facts: Dict[str, Any] = {}
        ok = 0
        for r, text in zip(recs, outs):
            obj = extract_json(text)
            if isinstance(obj, dict) and any(k in obj for k in prompts.EXTRACT_KEYS):
                facts[r.id] = obj
                ok += 1
        if progress:
            print(f"  [사실추출] 성공 {ok}/{len(recs)}건 · {time.time() - t0:.0f}s")
        self.stats.n_extracted = ok
        return facts

    def render(self, task: Task) -> Task:
        budget = task.group.budget
        while True:
            msgs = prompts.build_messages(
                task.rec, task.group, task.items, self.tbl,
                gosi=self.gosi,
                law_text=self.law_texts.get(task.group.key, ""),
                budget=budget,
                graded=self.graded,
                select=self.select,
                dual=self.dual,
                facts=self.facts.get(task.rec.id) if self.extract else None,
                lang=self.lang,
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
        if self.extract:
            self.facts = self._extract(recs, chunk=chunk, progress=progress)
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

            def _cfg():
                return GenConfig(max_tokens=self.max_tokens, seed=self.seed,
                                 schema=prompts.build_schema(
                                     sig, graded=self.graded,
                                     select=self.select, dual=self.dual))

            cfg = _cfg()
            for s in range(0, len(indices), chunk):
                part = indices[s:s + chunk]
                if skippable:
                    left = self._time_left()
                    if left is not None:
                        per = (time.time() - t0) / max(1, done)
                        if left <= per * len(part) * 1.3 and self.dual:
                            # 양면 판단은 출력이 기존의 2.25배다(캐시 실측 154→347자).
                            # 시간이 빡빡해지면 보강 호출을 통째로 버리는 것보다
                            # 양면 판단만 끄고 계속하는 쪽이 낫다 — 버린 칸은 0 이 되지만
                            # 짧은 판정이라도 받으면 그 칸의 재현율이 살아난다.
                            print(f"  ⏱ 시간예산 압박 (남은 {left:.0f}s) → "
                                  f"양면 판단 끄고 계속")
                            self.dual = False
                            cfg = _cfg()
                            per *= 0.7   # 출력이 짧아진 만큼 청크당 소요도 줄어든다
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
                              schema=prompts.build_schema(tuple(tasks[i].items), graded=self.graded,
                                                          select=self.select, dual=self.dual))
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
            if self.select:
                parsed, missing = parse_select(text, t.items)
            else:
                parsed, missing = parse_group(text, t.items, graded=self.graded)
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
                # 문턱을 넘긴 칸만 재검토 대상이다 (등급은 원본 그대로 보존된다)
                if (int(cell.get(prompts.GRADE_KEY, 0) or 0) >= self.grade_threshold
                        and g.get(item, True)):
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
                 dropped: Optional[set] = None,
                 threshold: Optional[int] = None) -> Dict[str, Dict[str, Any]]:
        """LLM 원본 등급 + 게이팅 + 규칙을 결합해 **최종 0/1** 24항목을 만든다.

        여기가 **원본 등급이 0/1 로 바뀌는 유일한 지점**이다.
        `threshold` 를 주면 그 값으로, 안 주면 `self.grade_threshold` 로 판정한다 —
        저장된 등급을 재채점할 때 이 인자만 바꾸면 되므로 API 재호출이 필요 없다.
        """
        th = self.grade_threshold if threshold is None else threshold
        g = gating.gate(rec)
        src = rec.full_text
        out: Dict[str, Dict[str, Any]] = {}

        rule_hint = self._rule_hints(rec) if self.use_rules else {}

        for v in ITEMS:
            if not g[v]:
                out[v] = {"위반여부": 0, "근거문구": ""}
                continue
            cell = judged.get(v) or {}
            # 문턱 적용 — LLM 원본 등급은 judged 안에 그대로 남는다.
            hit = 1 if int(cell.get(prompts.GRADE_KEY, 0) or 0) >= th else 0
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

        # v21 — 공고가 정한 구성원별 최소지분율을 조문 기준(지방 5% / 국가 10%)과 견준다.
        # dev 200건에서 규칙 TP 6 · FP 0 · FN 0 (F1 1.000) vs LLM F1 0.500 →
        # 규칙이 확실히 나으므로 강제한다(v23 과 같은 근거).
        # 지분율 표기가 없거나 분담이행 전용이면 None 을 돌려 LLM 에 맡긴다.
        v21, _ = compare.check_v21(rec)
        if v21 is not None:
            hints["v21"] = 1 if v21 else 0

        # ⛔ v24 는 `compare.check_v24_positive` 를 만들었지만 **배선하지 않는다.**
        #    dev 200 실측:
        #      LLM 단독  TP 2 FP 12 FN 6 → F1 0.182
        #      규칙 단독  TP 2 FP  2 FN 6 → F1 0.333   ← 규칙이 더 낫다
        #      합집합    TP 2 FP 15 FN 6 → F1 0.160   ← **LLM 보다 나쁘다**
        #    규칙이 맞춘 2건(PPS-DEV-29·057)이 LLM 이 이미 맞춘 것과 **같은 레코드**다.
        #    그래서 양성 전용 힌트로 합쳐도 새 TP 는 0 이고 FP 만 늘어난다.
        #    규칙 단독으로 넘기면 dev 에서는 오르지만 양성 8건 표본이라
        #    09-07(세부품명 게이팅, 양성 19건 → 리더보드 −0.105)과 같은 도박이다.
        #    → 리더보드로 검증할 여력이 생길 때까지 넣지 않는다.
        return hints
