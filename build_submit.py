#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""submit.zip 생성 — pps/ 패키지를 자기완결 script.py 한 개로 묶는다.

왜 한 파일인가
  제출 규약은 zip 루트의 script.py 를 실행한다. 평가 서버가 "폴더 구조 정규화"를
  수행하므로 여러 모듈을 흩어 두면 import 경로가 깨질 위험이 있다.
  모듈 소스를 문자열로 심고 실행 시점에 sys.modules 에 올리면
  네임스페이스를 그대로 유지하면서 한 파일로 만들 수 있다.
  (소스를 이어붙이는 방식은 모듈 간 동명 함수가 서로를 덮어써서 위험하다 —
   실제로 records.validate 와 submission.validate 가 충돌한다.)

제출물에서 반드시 빠져야 하는 것
  · ApiRunner / RateLimiter — 외부 API 호출. 규칙 위반(실격) 사유.
    runner.py 의 DEV ONLY 마커 구간을 제거하고, 제거되었는지 검증한다.

사용:
    python3 build_submit.py                 # submit.zip 생성 + 자체 점검
    python3 build_submit.py --smoke         # 생성 후 --mock 으로 실제 실행까지
"""
from __future__ import annotations

import argparse
import base64
import io
import os
import re
import subprocess
import sys
import tempfile
import zipfile

ROOT = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.join(ROOT, "pps")

# 의존 순서 (앞이 먼저 로드된다)
MODULES = [
    "law", "records", "evidence", "presence", "pumnum", "gosimatch", "compare", "spec", "schedule",
    "sections", "gating", "prompts", "runner", "pipeline", "submission",
]

DEV_BLOCK = re.compile(
    r"^# ===== BEGIN DEV ONLY =+\s*$.*?^# ===== END DEV ONLY =+\s*$\n",
    re.M | re.S,
)

FORBIDDEN = [
    ("ApiRunner", "외부 API 러너"),
    ("urllib.request", "외부 HTTP 호출"),
    ("PPS_API_KEY", "외부 API 키"),
    ("chat/completions", "외부 엔드포인트"),
]

HEADER = '''#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""나라장터 자체입찰 공고 법령 위반사항 모니터링 — 추론 코드.

평가 서버가 이 파일을 `python script.py` 로 실행한다.
  입력  $PPS_DATA_DIR/test.jsonl.gz (+ 항목표.json · 법령패키지 · 중기부고시)
  출력  $PPS_OUTPUT_DIR/submission.csv  (49열: id, v1..v24, e1..e24)
  모델  $PPS_MODEL_DIR 의 고정 LLM 을 vLLM offline API 로 인프로세스 로드

구조
  ① 메타데이터 게이팅 — 법령상 위반이 성립할 수 없는 항목을 0으로 확정
  ② 섹션 압축      — 판정에 필요한 절만 뽑아 프롬프트 밀도를 올림
  ③ 그룹별 LLM 판정 — 24항목을 4개 묶음으로 나눠 각각 질의
  ④ 규칙 결합·근거 정합 — 전체 원문 스캔 결과로 보정, 근거문구를 원문에 맞춤

이 파일은 build_submit.py 가 pps/ 패키지에서 자동 생성한다. 직접 수정하지 말 것.
"""
from __future__ import annotations

import argparse
import base64
import os
import sys
import time
import types

# 모듈 소스는 base64 로 심는다 — 원본에 삼중따옴표·백슬래시가 있어도 안전하다.
_MODULE_ORDER = {order!r}
_SOURCES = {{}}


def _install_modules() -> None:
    """심어 둔 모듈 소스를 sys.modules 에 올린다 (네임스페이스 보존).

    소스를 이어붙이지 않는 이유: 모듈 간 동명 함수가 서로를 덮어쓴다
    (records.validate 와 submission.validate 가 실제로 충돌한다).
    """
    pkg = types.ModuleType("pps")
    pkg.__path__ = []          # 패키지로 인식시켜 상대 import 를 허용
    sys.modules["pps"] = pkg
    for short in _MODULE_ORDER:
        name = "pps." + short
        mod = types.ModuleType(name)
        mod.__package__ = "pps"
        sys.modules[name] = mod
        src = base64.b64decode(_SOURCES[short]).decode("utf-8")
        exec(compile(src, name, "exec"), mod.__dict__)
        setattr(pkg, short, mod)
'''

MAIN = '''

def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mock", action="store_true", help="모델 없이 배선·형식만 확인")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--max-model-len", type=int, default=16384)
    # 1200 으로 충분하다. 캐시 실측에서 정상 응답 최대가 580자였다 —
    # 상한에 닿은 적이 없다. 자세한 근거는 pps/pipeline.py 의 max_tokens 주석 참조.
    ap.add_argument("--max-tokens", type=int, default=1200)
    ap.add_argument("--quant", default="int8_per_channel_weight_only")
    ap.add_argument("--verify", action="store_true",
                    help="2단계 검증 — dev200 측정에서 상한을 0.484→0.421 로 깎았다. "
                         "취소 80건 중 진짜양성이 28건이라 재현율 손실이 크다. 기본 비활성.")
    ap.add_argument("--time-budget", type=float,
                    default=float(os.environ.get("PPS_TIME_BUDGET", 6300)),
                    help="초. 프로세스 시작 기준. 소진되면 남은 호출을 포기하고 "
                         "그때까지 판정한 결과로 제출 파일을 낸다. "
                         "평가 서버 제한은 2시간(7200s)이며 기본값은 안전여유 900s.")
    args = ap.parse_args(argv)
    _t_start = time.monotonic()

    _install_modules()
    from pps import prompts, pumnum, submission          # noqa: F401
    from pps.pipeline import Pipeline
    from pps.records import ITEMS, load_item_table, load_records
    from pps.runner import make_runner

    t0 = time.time()
    data_dir = os.environ.get("PPS_DATA_DIR", "./data")
    out_dir = os.environ.get("PPS_OUTPUT_DIR", "./output")
    model_dir = os.environ.get("PPS_MODEL_DIR", "")
    out_path = os.path.join(out_dir, "submission.csv")

    def log(m):
        print(f"[pps] {m}", file=sys.stderr, flush=True)

    log(f"data={data_dir} out={out_dir} model={model_dir}")

    recs = load_records(os.path.join(data_dir, "test.jsonl.gz"), limit=args.limit)
    ids = [r.id for r in recs]
    log(f"{len(recs)}건 로드 ({time.time() - t0:.0f}s)")

    # 어떤 실패에도 형식에 맞는 제출 파일은 남긴다 — 빈 출력이 제출 오류보다 낫다
    submission.write_csv([submission.empty_row(i) for i in ids], out_path)

    tbl = load_item_table(data_dir)
    try:
        gosi = pumnum.load_gosi(data_dir)
        log(f"중기부고시 세부품명 {len(gosi.by_num)}개")
    except Exception as e:                                  # noqa: BLE001
        log(f"고시 로드 실패({type(e).__name__}) — 경쟁제품 대조 없이 진행")
        gosi = None

    if args.mock:
        runner = make_runner("mock", items=ITEMS)
    else:
        runner = make_runner(
            "vllm", model_dir=model_dir, quantization=args.quant,
            max_model_len=args.max_model_len)
        log(f"모델 로드 {runner.load_seconds:.0f}s")

    # 모델 로드에 쓴 시간을 빼고 남은 예산을 추론에 배정한다.
    pipe = Pipeline(runner, tbl, gosi=gosi,
                    max_tokens=args.max_tokens,
                    prompt_budget=args.max_model_len - args.max_tokens,
                    deadline=_t_start + args.time_budget)
    log(f"추론 시간예산 {args.time_budget - (time.monotonic() - _t_start):.0f}s 남음")

    try:
        judged = pipe.run(recs, chunk=args.chunk, progress=True)
    except Exception as e:                                  # noqa: BLE001
        log(f"추론 실패 → 기본값 제출 유지: {type(e).__name__}: {e}")
        return 0

    dropped = {}
    if args.verify:
        try:
            dropped = pipe.verify(recs, judged, chunk=args.chunk, progress=True)
        except Exception as e:                              # noqa: BLE001
            log(f"2단계 검증 실패 → 1단계 판정 유지: {type(e).__name__}")

    rows = []
    for r in recs:
        try:
            rows.append(submission.to_row(
                r.id, pipe.finalize(r, judged.get(r.id, {}), dropped.get(r.id))))
        except Exception as e:                              # noqa: BLE001
            log(f"{r.id} 후처리 실패 → 기본값: {type(e).__name__}")
            rows.append(submission.empty_row(r.id))

    submission.write_csv(rows, out_path)
    errs = submission.validate(out_path, ids)
    log(pipe.stats.report().replace("\\n", " | "))
    log("자가검증 " + ("PASS" if not errs else f"FAIL {errs[:5]}"))
    log(f"총 {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def read_module(short: str) -> str:
    src = io.open(os.path.join(PKG, short + ".py"), encoding="utf-8").read()
    if short == "runner":
        src, n = DEV_BLOCK.subn("", src)
        if n != 1:
            raise SystemExit(
                f"runner.py 의 DEV ONLY 마커를 {n}개 찾았다 (1개여야 함). "
                "외부 API 코드가 제출물에 실릴 위험이 있어 중단한다.")
    return src


def build(out_zip: str) -> str:
    sources = {short: read_module(short) for short in MODULES}

    # 안전 점검 — 금지 요소가 모듈 소스에 남았는지 (base64 로 감싸기 전에 본다)
    joined = "\n".join(sources.values())
    for needle, why in FORBIDDEN:
        if needle in joined:
            raise SystemExit(
                f"❌ 제출물에 '{needle}' 이 남아 있다 ({why}). 중단한다.\n"
                f"   해당 코드는 runner.py 의 DEV ONLY 마커 안에 두어야 한다.")

    parts = [HEADER.format(order=MODULES)]
    for short in MODULES:
        b64 = base64.b64encode(sources[short].encode("utf-8")).decode("ascii")
        wrapped = "\n".join(b64[i:i + 96] for i in range(0, len(b64), 96))
        parts.append(f'\n_SOURCES[{short!r}] = (\n'
                     + "\n".join(f'    "{ln}"' for ln in wrapped.split("\n"))
                     + "\n)\n")
    parts.append(MAIN)
    script = "".join(parts)

    script_path = os.path.join(ROOT, "build", "script.py")
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    io.open(script_path, "w", encoding="utf-8").write(script)

    req_path = os.path.join(ROOT, "build", "requirements.txt")
    io.open(req_path, "w", encoding="utf-8").write(
        "# 평가 서버 기본 패키지만 사용합니다 (추가 설치 없음).\n"
        "# vllm·torch·transformers·xgrammar 는 서버 고정이므로 넣지 않습니다.\n")

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
        z.write(script_path, "script.py")
        z.write(req_path, "requirements.txt")
        model_dir = os.path.join(ROOT, "model")
        for root, _, files in os.walk(model_dir):
            for fn in files:
                if fn == ".DS_Store":
                    continue
                p = os.path.join(root, fn)
                z.write(p, os.path.join("model", os.path.relpath(p, model_dir)))
    return script_path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "submit.zip"))
    ap.add_argument("--smoke", action="store_true",
                    help="생성한 script.py 를 --mock 으로 실제 실행")
    args = ap.parse_args()

    script_path = build(args.out)
    size = os.path.getsize(args.out)
    with zipfile.ZipFile(args.out) as z:
        names = z.namelist()
    print(f"✅ {args.out} ({size:,} bytes)")
    print(f"   {names}")
    print(f"   script.py {os.path.getsize(script_path):,} bytes")

    if args.smoke:
        with tempfile.TemporaryDirectory() as td:
            env = dict(os.environ)
            env["PPS_DATA_DIR"] = os.path.join(ROOT, "open", "data")
            env["PPS_OUTPUT_DIR"] = td
            r = subprocess.run(
                [sys.executable, script_path, "--mock"],
                env=env, capture_output=True, text=True)
            print("\n--- smoke (--mock) ---")
            print(r.stderr.strip()[-2000:])
            out = os.path.join(td, "submission.csv")
            if r.returncode != 0:
                print(r.stdout[-2000:])
                return 1
            if not os.path.exists(out):
                print("❌ submission.csv 가 생성되지 않았다")
                return 1
            lines = io.open(out, encoding="utf-8").read().splitlines()
            print(f"✅ submission.csv {len(lines) - 1}행 · "
                  f"{len(lines[0].split(','))}열")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
