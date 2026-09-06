# -*- coding: utf-8 -*-
"""LLM 러너 추상화.

세 가지 구현이 같은 인터페이스를 공유한다.

  MockRunner   모델 없이 배선·형식만 확인 (CI·빠른 스모크)
  VllmRunner   제출용. PPS_MODEL_DIR 의 고정 모델을 vLLM offline API로 인프로세스 로드.
  (개발용 원격 러너)  아래 DEV ONLY 구간에 있다. OpenAI 호환 엔드포인트로
               gemma-4-31b-it 등을 불러 프롬프트를 다듬는 용도다.
               ⚠️ 제출물에는 절대 포함되지 않는다 — 대회 규칙상 추론 코드에서의
                  외부 호출은 금지(실격)다. build_submit.py 가 해당 구간을 제거하고,
                  관련 식별자가 남아 있으면 빌드를 중단한다.

프롬프트·파싱·후처리는 러너와 무관하게 동일하게 동작해야 한다.
그래야 API에서 튜닝한 결과가 평가 서버로 그대로 넘어간다.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

Messages = List[Dict[str, str]]

# 개발 전용 러너 등록소. 제출물에서는 DEV ONLY 구간이 통째로 제거되므로 비어 있다.
# build_submit.py 가 소스에서 외부 호출 관련 식별자가 남았는지 검사한다.
_EXTRA_RUNNERS: Dict[str, Any] = {}


@dataclass
class GenConfig:
    temperature: float = 0.0
    max_tokens: int = 1536
    seed: int = 20260826
    schema: Optional[Dict[str, Any]] = None      # JSON Schema 제약 디코딩


# --------------------------------------------------------------------------- Mock

class MockRunner:
    """전 항목 0 + 근거 null 을 돌려준다. 형식 검증용."""

    name = "mock"
    load_seconds = 0.0
    per_request_schema = True     # 요청마다 다른 스키마 허용 → 파이프라인이 한 풀로 병렬 실행

    def __init__(self, items: Sequence[str] = (), **_: Any):
        self.items = list(items) or [f"v{i}" for i in range(1, 25)]

    def count_tokens(self, messages: Messages) -> int:
        return sum(len(m["content"]) for m in messages) // 2   # 한국어 ≈ 2자/토큰

    def generate(self, batch: List[Messages], cfg: GenConfig) -> List[str]:
        payload = {v: {"위반여부": 0, "근거문구": None} for v in self.items}
        return [json.dumps(payload, ensure_ascii=False) for _ in batch]

    def generate_mixed(self, batch: List[Messages], cfgs: List[GenConfig],
                       on_done=None) -> List[str]:
        out = []
        for i, cfg in enumerate(cfgs):
            items = list((cfg.schema or {}).get("required") or self.items)
            out.append(json.dumps({v: {"위반여부": 0, "근거문구": None} for v in items},
                                  ensure_ascii=False))
            if on_done:
                on_done(i)
        return out


# ===== BEGIN DEV ONLY =====================================================
# 이 구간은 build_submit.py 가 제출물에서 통째로 제거한다.
# 대회 규칙: "제출되는 추론 코드에서의 외부 LLM·외부 API 호출" 금지(실격).
# ⚠️ 이 마커 안에는 개발 전용 코드만 둔다. 제출에 필요한 클래스를 넣으면
#    빌드 시 함께 삭제되어 평가 서버에서 NameError 로 죽는다(실제로 겪었다).

class RateLimiter:
    """분당 호출 수 제한."""

    def __init__(self, per_minute: int):
        self.interval = 60.0 / max(1, per_minute)
        self._lock = threading.Lock()
        self._next = 0.0

    def acquire(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait = self._next - now
            if wait > 0:
                time.sleep(wait)
                now = time.monotonic()
            self._next = max(now, self._next) + self.interval


class ApiRunner:
    """OpenAI 호환 Chat Completions 엔드포인트 (개발용).

    환경변수
      PPS_API_BASE   예: https://integrate.api.nvidia.com/v1   (NVIDIA NIM)
      PPS_API_KEY    Bearer 토큰
      PPS_API_MODEL  예: google/gemma-4-31b-it
      PPS_API_RPM    분당 호출 상한 (기본 40)

    NIM 주의점
      · thinking 모드는 끈다. 제약 디코딩과 충돌하고 토큰 예산만 잡아먹는다.
        평가 서버에서는 JSON Schema 로 출력이 고정되므로 사고 흔적을 낼 자리가 없다.
      · temperature 는 0. 평가 서버가 temperature 0 + seed 고정으로 돌기 때문에
        개발 단계에서 샘플링을 켜면 프롬프트 개선인지 운인지 구분할 수 없다.
      · response_format=json_schema 미지원이면 자동으로 끄고 프롬프트 지시 +
        후처리 파싱으로 대체한다(파싱 경로는 vLLM 과 동일하다).
    """

    name = "api"
    load_seconds = 0.0
    per_request_schema = True     # HTTP 요청마다 스키마가 따로 가므로 한 풀에서 병렬 가능

    # 디스크 응답 캐시 — 개발 반복의 핵심.
    # NIM 은 건당 25~66s 라 dev 40건(160콜)에 22분이 걸린다. 그 속도로는
    # "무엇이 이득이고 무엇이 손해인지" 원인을 분리할 수 없다.
    # 프롬프트가 그대로면 응답을 재사용하고, 규칙·후처리만 바꾼 재측정은 즉시 끝난다.
    cache_dir: Optional[str] = None

    def __init__(
        self,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        rpm: Optional[int] = None,
        max_workers: int = 8,
        use_response_format: bool = True,
        **_: Any,
    ):
        self.model = model or os.environ.get("PPS_API_MODEL", "google/gemma-4-31b-it")
        self.base_url = (base_url or os.environ.get("PPS_API_BASE", "")).rstrip("/")
        self.api_key = api_key or os.environ.get("PPS_API_KEY", "")
        self.limiter = RateLimiter(int(rpm or os.environ.get("PPS_API_RPM", 40)))
        self.max_workers = max_workers
        self.use_response_format = use_response_format
        self.cache_dir = os.environ.get("PPS_API_CACHE", ".cache/api")
        self.cache_hits = 0
        self.cache_misses = 0
        if self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)
        if not self.base_url or not self.api_key:
            raise RuntimeError(
                "ApiRunner 에는 PPS_API_BASE 와 PPS_API_KEY 가 필요합니다.")
        self._tok = None

    # ---- 토큰 추정 -------------------------------------------------------
    def count_tokens(self, messages: Messages) -> int:
        """정확한 토크나이저가 있으면 쓰고, 없으면 보수적 추정."""
        if self._tok is None:
            self._tok = self._load_tokenizer()
        text = "\n".join(m["content"] for m in messages)
        if self._tok is False:
            return len(text) // 2 + 64
        return len(self._tok.encode(text)) + 64

    def _load_tokenizer(self):
        path = os.environ.get("PPS_TOKENIZER_DIR")
        if not path:
            return False
        try:
            from transformers import AutoTokenizer
            return AutoTokenizer.from_pretrained(path)
        except Exception:
            return False

    # ---- 호출 -----------------------------------------------------------
    def _cache_key(self, messages: Messages, cfg: GenConfig) -> str:
        import hashlib
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "schema": cfg.schema,
        }, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def _cache_get(self, key: str) -> Optional[str]:
        if not self.cache_dir:
            return None
        p = os.path.join(self.cache_dir, key + ".txt")
        if not os.path.exists(p):
            return None
        with open(p, encoding="utf-8") as f:
            return f.read()

    def _cache_put(self, key: str, text: str) -> None:
        if not self.cache_dir or not text:
            return
        p = os.path.join(self.cache_dir, key + ".txt")
        tmp = p + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(text)
        os.replace(tmp, p)

    def _one(self, messages: Messages, cfg: GenConfig, attempt: int = 0) -> str:
        if attempt == 0:
            key = self._cache_key(messages, cfg)
            hit = self._cache_get(key)
            if hit is not None:
                self.cache_hits += 1
                return hit
            self.cache_misses += 1
        import urllib.error
        import urllib.request

        body: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": cfg.temperature,
            "max_tokens": cfg.max_tokens,
            "stream": False,
            # Gemma 4 계열은 사고 모드를 켤 수 있다. 여기서는 끈다 —
            # 평가 서버는 JSON Schema 로 출력이 고정되어 사고 흔적을 낼 자리가 없다.
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if cfg.temperature > 0:
            body["top_p"] = 0.95
        if cfg.schema is not None and self.use_response_format:
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "판정", "schema": cfg.schema, "strict": True},
            }

        req = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        self.limiter.acquire()
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            text = data["choices"][0]["message"]["content"] or ""
            self._cache_put(self._cache_key(messages, cfg), text)
            return text
        except urllib.error.HTTPError as e:
            detail = e.read().decode("utf-8", "replace")[:300]
            # 429/5xx 는 지수 백오프 재시도
            if e.code in (408, 409, 429, 500, 502, 503, 504) and attempt < 5:
                time.sleep(min(60, (2 ** attempt) + random.random()))
                return self._one(messages, cfg, attempt + 1)
            # json_schema 미지원이면 한 번만 끄고 재시도
            if e.code == 400 and self.use_response_format and cfg.schema is not None:
                self.use_response_format = False
                return self._one(messages, cfg, attempt)
            raise RuntimeError(f"API {e.code}: {detail}") from e
        except Exception:
            if attempt < 5:
                time.sleep(min(60, (2 ** attempt) + random.random()))
                return self._one(messages, cfg, attempt + 1)
            raise

    def generate(self, batch: List[Messages], cfg: GenConfig) -> List[str]:
        """레이트리밋 안에서 스레드 병렬 호출. 실패 건은 빈 문자열."""
        return self.generate_mixed(batch, [cfg] * len(batch))

    def generate_mixed(self, batch: List[Messages], cfgs: List[GenConfig],
                       on_done=None) -> List[str]:
        """요청마다 다른 GenConfig(스키마) — 전체를 한 풀에서 병렬 실행.

        스키마별로 배치를 쪼개 순차 실행하면 워커가 놀아서 dev 20건에 10분이 걸렸다.
        HTTP 는 요청마다 스키마를 따로 보내므로 굳이 묶을 이유가 없다.
        """
        from concurrent.futures import ThreadPoolExecutor

        out: List[str] = [""] * len(batch)

        def work(i: int) -> None:
            try:
                out[i] = self._one(batch[i], cfgs[i])
            except Exception as e:                       # noqa: BLE001
                print(f"  ! API 실패 [{i}] {type(e).__name__}: {str(e)[:160]}")
                out[i] = ""
            if on_done:
                on_done(i)

        with ThreadPoolExecutor(max_workers=self.max_workers) as ex:
            list(ex.map(work, range(len(batch))))
        return out


_EXTRA_RUNNERS["api"] = ApiRunner
# ===== END DEV ONLY =======================================================


# --------------------------------------------------------------------------- vLLM (제출용)

class VllmRunner:
    """평가 서버의 고정 모델을 vLLM offline API로 로드한다.

    ⚠️ vLLM 이 하위 프로세스를 spawn 하므로 반드시 `if __name__ == "__main__":`
       아래에서 생성해야 한다. 모듈 최상위에서 만들면 즉시 실패한다.
    """

    name = "vllm"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        quantization: str = "int8_per_channel_weight_only",
        max_model_len: int = 16384,
        gpu_memory_utilization: float = 0.92,
        seed: int = 20260826,
        tensor_parallel_size: int = 1,
        **_: Any,
    ):
        t0 = time.time()
        from vllm import LLM

        self.model_dir = model_dir or os.environ.get("PPS_MODEL_DIR", "")
        kw: Dict[str, Any] = dict(
            model=self.model_dir,
            tokenizer=self.model_dir,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            seed=seed,
            tensor_parallel_size=tensor_parallel_size,
            dtype="auto",
        )
        if quantization:
            kw["quantization"] = quantization
        self.llm = LLM(**kw)
        self.tok = self.llm.get_tokenizer()
        self.max_model_len = max_model_len
        self.load_seconds = time.time() - t0

    def count_tokens(self, messages: Messages) -> int:
        try:
            ids = self.tok.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True)
            if hasattr(ids, "keys") and "input_ids" in ids:
                ids = ids["input_ids"]
            return len(ids)
        except Exception:
            return len(self.tok.encode("\n".join(m["content"] for m in messages)))

    def generate(self, batch: List[Messages], cfg: GenConfig) -> List[str]:
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        sp_kw: Dict[str, Any] = dict(
            temperature=cfg.temperature, max_tokens=cfg.max_tokens, seed=cfg.seed)
        if cfg.schema is not None:
            sp_kw["structured_outputs"] = StructuredOutputsParams(
                json=cfg.schema, disable_any_whitespace=True)
        sp = SamplingParams(**sp_kw)
        outs = self.llm.chat(batch, sampling_params=sp, use_tqdm=False)
        return [o.outputs[0].text if o.outputs else "" for o in outs]


# --------------------------------------------------------------------------- 팩토리

def make_runner(kind: str, **kw: Any):
    kind = (kind or "mock").lower()
    if kind == "mock":
        return MockRunner(**kw)
    if kind == "vllm":
        return VllmRunner(**kw)
    if kind in _EXTRA_RUNNERS:
        return _EXTRA_RUNNERS[kind](**kw)
    raise ValueError(
        f"알 수 없는 러너: {kind} (사용 가능: mock, vllm"
        + (", " + ", ".join(_EXTRA_RUNNERS) if _EXTRA_RUNNERS else "")
        + "). 개발 전용 러너는 제출물에서 제거된다.")


def run_with_fallback(runner, batch: List[Messages], cfg: GenConfig) -> List[str]:
    """배치가 통째로 실패하면 건 단위로 재시도하고, 그래도 실패하면 빈 출력."""
    try:
        return runner.generate(batch, cfg)
    except Exception as e:                                  # noqa: BLE001
        print(f"  ! 배치({len(batch)}건) 실패 → 건 단위 재시도: "
              f"{type(e).__name__}: {str(e)[:160]}")
    out = []
    for m in batch:
        try:
            out.append(runner.generate([m], cfg)[0])
        except Exception as e:                              # noqa: BLE001
            print(f"  ! 건 단위 실패 → 빈 출력: {type(e).__name__}: {str(e)[:160]}")
            out.append("")
    return out
