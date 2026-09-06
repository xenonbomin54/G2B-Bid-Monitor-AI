# 진행 상황 — 2026-09-06

## 지금 상태: 파이프라인·안전장치 완성. **LLM 품질은 아직 한 번도 측정 안 됨** ← 최우선

`submit.zip`이 빌드되고 생성된 `script.py`가 실제로 실행되어 49열 제출 파일을 만든다.
규칙 준수 안전장치까지 끝났다. 남은 건 판정 품질이다.

### ⚠️ 다음 세션이 가장 먼저 알아야 할 것

**API 호출이 예상보다 훨씬 느리다.** dev 20건(80콜)에 10분이 지나도 안 끝나서 중단했다.
예상은 2~3분이었다. 원인 미규명 — 아래 중 하나로 보인다.

1. 그룹별 프롬프트가 커서(4,000자 문서 + 지침 + 사실블록) 생성이 오래 걸림
2. `chat_template_kwargs.enable_thinking=False` 가 NIM에서 무시되어
   사고 토큰을 max_tokens(1200)까지 뽑고 있음
3. 429 재시도 백오프가 도는 중 (최대 5회, 최대 60s 대기)

**진단 방법**: 파이프를 통하지 말고(`| tail` 금지 — 출력이 버퍼링되어 진행이 안 보인다)
그룹 1건씩 지연시간을 직접 재라.

```bash
source .env
python3 - <<'PY'
import sys, time; sys.path.insert(0,'.')
from pps.runner import make_runner, GenConfig
from pps.records import load_item_table, load_records
from pps import prompts, pumnum, gating
r = make_runner("api", max_workers=4)
rec = load_records('open/dev.jsonl.gz', limit=1)[0]
tbl = load_item_table('open/data'); gosi = pumnum.load_gosi('open/data')
g = gating.gate(rec)
for grp in prompts.GROUPS:
    items = [i for i in grp.items if g[i]]
    if not items: continue
    msgs = prompts.build_messages(rec, grp, items, tbl, gosi=gosi)
    t = time.time()
    out = r.generate([msgs], GenConfig(max_tokens=1200, schema=prompts.build_schema(items)))
    print(f"{grp.key:<14} {len(items):>2}항목 ~{r.count_tokens(msgs):>6,}토큰 "
          f"{time.time()-t:>6.1f}s 출력{len(out[0]):>5}자", flush=True)
PY
```

느린 원인이 (2)라면 `max_tokens` 를 400으로 줄이고 프롬프트에서 사고를 금지한다.
(1)이라면 그룹 예산을 4,000 → 2,500자로 낮춘다 (섹션 파서 측정상 3,000자에서
근거 포함률 90.7%라 손실이 작다).

---

## 바로 실행할 수 있는 것

```bash
# 환경변수 (개발 전용 — 제출물엔 안 들어감)
source .env            # .env.example 복사해서 키 채우기

python3 tools/audit_gates.py       # 게이팅 안전성 감사
python3 tools/probe_sections.py    # 섹션 파서 근거 포함률
python3 tools/probe_presence.py    # 부재탐지 규칙 성능
python3 tools/evaluate.py --runner mock --limit 40    # 배선 확인
python3 tools/evaluate.py --runner api --limit 20     # ★ 다음 할 일
python3 build_submit.py --smoke    # submit.zip 생성 + 실행 검증
```

---

## 측정된 것 (dev 200건, 전부 재현 가능)

| 항목 | 결과 |
|---|---|
| **게이팅** | 4,800칸 중 **1,313칸(27.4%) 0으로 확정, 죽인 진짜 양성 0건** |
| **섹션 파서** | 3,000자 예산에서 근거 포함률 **90.7%** (베이스라인 79.6%) → 토큰 4배 절감 |
| **부재탐지 규칙 단독** | 평균 F1 0.21 — LLM 결합 필요 (§미해결) |
| **파이프라인** | 공고당 LLM 호출 **4.00회** · Mock 전 구간 통과 |
| **API** | NVIDIA NIM, `json_schema` 지원 확인, 단건 14.8s |
| **워치독** | 마감 경과 상태에서도 공고당 1회 호출 보장 + 유효 CSV 생성 확인 |

### 규칙 2-1 준수 구조 (2026-09-06 수정)
> 각 공고에 대해 고정 LLM 정상 호출 1회 이상. 없으면 **제출물 전체가 무효 처리**.

워치독이 남은 호출을 건너뛰면 이 규칙을 깨뜨릴 수 있었다. 구조를 둘로 나눴다.

- **1단계 필수** — 공고당 첫 작업(`Task.mandatory`). 시간이 없어도 **건너뛰지 않는다**.
  v1의 게이트가 `_always` 라 모든 공고가 G1 작업을 최소 하나 갖는다.
- **2단계 보강** — 나머지. 시간 부족 시 생략, 해당 항목은 0으로 제출.

실패 모드 우선순위: **무효 처리(최악) > 제출 오류 > 일부 항목 0점(감수)**.
매 실행마다 `records_without_call` 을 세어 0이 아니면 리포트 맨 위에 ❌ 로 띄운다.

### 게이팅이 법 구조에서 나온 것임을 확인
국가 시행령 제21조①10호 / 지방 시행령 제20조①12호가 금액 구간별 허용 제한을 규정한다.
dev 양성 분포가 이 구간과 100% 일치했다 — 상관이 아니라 법 그 자체였다.

| 추정가격 | 허용되는 제한 | 항목 |
|---|---|---|
| 1억 미만 | 소기업·소상공인·벤처·창업기업 | v17 v18 |
| 1억~고시금액 | 중소기업자 | v15 v16 |
| 고시금액 이상 | 제한 불가 | v14 |

---

## 다음에 할 일 (우선순위)

### 0. API 지연시간 원인 규명 ★ (위 경고 참조)
이게 안 풀리면 반복 루프가 성립하지 않는다. 나머지 전부가 여기 막혀 있다.

### 1. LLM 품질 첫 측정 ★
```bash
python3 tools/evaluate.py --runner api --limit 20 --save out/dev20_pred.csv
python3 tools/analyze_errors.py out/dev20_pred.csv
```
dev 20건으로 Macro F1 기준선. 그 다음 전체 200건.
**이 숫자가 나와야 그 다음 판단이 가능하다.** 지금은 아무것도 모르는 상태다.

`analyze_errors.py` 가 실패를 원인별로 갈라 준다:
게이팅차단(법 해석 오류·최우선) / 입력누락(섹션 파서) / 규칙보정 / 모델오판(프롬프트).

### 2. 24항목 판정 기준서 (Fable 할당량 쓸 1순위)
`prompts.py` 의 그룹별 `지시` 는 조문과 dev 예시를 훑어 쓴 초안이다.
법령패키지를 정독하고 dev 양성 54건을 전부 대조해 정밀 규칙으로 다시 쓴다.
**두 군데에 동시에 쓰인다**: 모델 지시문 + 자가 라벨링 채점 기준.
dev 라벨로 검증되므로 틀려도 잡힌다.

### 3. 법령 조문 주입 (`law_texts`) — 보류 판단 필요
`prompts.build_messages` 의 `law_text` 인자가 비어 있다.
다만 조문 원문은 길고(제21조 하나가 수천 자), 고정 모델은 긴 문맥에 약하다(MRCR 44.1).
**판정 지침이 이미 법 논리를 압축해 담고 있으므로, 원문 주입이 이득인지 먼저 측정할 것.**
넣는다면 항목표.json의 항목→조문 고정 매핑을 쓴다(BM25 검색보다 정확).

### 4. 시간 예산 실측 (제출 전 필수)
공고당 4회 × 1,853건 = **7,412 콜**. L40S에서 2시간 안에 들어가는지 미확인.
안 되면 그룹 병합(4→3 또는 2)으로 줄인다.
워치독은 배선 완료 — `PPS_TIME_BUDGET` 기본 6300s.

### 4. 미해결: 중기간 경쟁제품 판별 (v10·v11·v12·v13 = 점수 16.7%)
- `meta.세부품명번호목록`은 dev 양성 25건 중 **0건** 매칭 → 메타 단독 불가
- 본문 품명번호 매칭 재현율 14~67%
- `meta.조항호내용`에 `"중소벤처기업부장관이 지정 공고한 물품"`(영 21①8호 문구)이
  들어있는 경우가 있음 → 강한 신호지만 확정 게이트는 아님
- 현재 설계: 규칙이 후보만 제시(`pumnum.candidate_lines`), LLM이 판정

---

## 설계 근거 메모

**왜 24항목 한 방 프롬프트를 버렸나**
고정 모델 gemma-4-26B-A4B-it는 개발용 31B보다 **추론력은 대등**하지만
(AIME 88.3 vs 89.2, GPQA 82.3 vs 84.3) **긴 문맥 탐색과 다단계 합성이 약하다**
(MRCR 44.1 vs 66.4, BBEH 64.8 vs 74.4, Tau2 68.2 vs 76.9).
→ 판단 난이도는 유지하고 문맥 길이·동시 항목 수만 줄인다.
→ "어려운 판단 하나를 짧은 텍스트로 묻기"가 최적점.
31B에서 잘 된다고 그대로 믿으면 안 된다.

**규칙 상수 튜닝은 되지만 모델 학습은 안 됨**
대회 규칙: 참가자가 학습시킨 가중치(분류기·회귀 포함)를 판정에 쓰면 실격.
임계값은 조문에서 읽어온 상수여야 하고, dev로 조정한 값은 `law.py`에
`UNCERTAIN` 주석과 근거를 남긴다.

---

## 제출 전 체크리스트

- [x] `script.py`가 `if __name__ == "__main__":` 아래에서 vLLM 생성
- [x] 절대경로 하드코딩 없음 (`PPS_*` 환경변수만)
- [x] 외부 API 코드 제출물에서 제거 — `build_submit.py`가 식별자 잔존 시 빌드 중단
- [x] 실패해도 형식 맞는 `submission.csv` 남김
- [x] 49열 · id 유일 · v∈{0,1} · e 500자 · 부재탐지 공란 자가검증
- [x] 공고당 LLM 호출 1회 이상 보장 (전 항목 게이팅돼도 강제 호출)
- [ ] **2시간 실행시간 검증** ← 미완
- [ ] 공고 단위 독립 예측 재확인 (전역 통계 사용 금지)
