# -*- coding: utf-8 -*-
"""LLM 원본 위반등급 저장·재로딩 — API 재호출 없이 문턱을 바꿔 재채점하기 위한 것.

왜 필요한가
  판정 문턱을 이진 0/1 로 받으면 문턱이 모델 안에 숨는다. 문턱을 하나 바꿔 보려면
  dev 200건 × 4그룹 = **API 800건을 다시 호출**해야 하고, 그 API 는 불안정해서
  한 번 돌리는 데 85분이 걸리고 10%가 끊긴다.
  등급(0~3)을 원본 그대로 저장해 두면 **호출은 한 번**이고 이후 문턱 스윕은 전부 로컬이다.

무엇을 저장하나
  record × 항목(v1~v24) 단위로 `위반등급` 과 `근거문구` 만 저장한다.
  게이팅·규칙·근거정합은 저장하지 않는다 — 그건 코드가 결정론적으로 다시 만든다
  (`Pipeline.finalize`). 즉 저장물은 **LLM 출력만** 담고, 재채점은 현재 코드로 한다.
  덕분에 규칙을 고친 뒤에도 같은 저장물로 재채점해 효과를 견줄 수 있다.

형식 (JSON)
  {
    "meta": {"판정모드": "graded"|"binary", "문턱": 2, "레코드수": 200, ...},
    "grades": {"<공고id>": {"v1": {"위반등급": 0, "근거문구": null}, ...}, ...}
  }

사용
    python3 tools/evaluate.py --runner api --graded --save-grades out/g.json
    python3 tools/sweep_threshold.py out/g.json          # API 호출 없음
"""
from __future__ import annotations

import io
import json
import os
from typing import Any, Dict, Optional

from .prompts import GRADE_KEY

SCHEMA_VERSION = 1


def save(path: str,
         grades: Dict[str, Dict[str, Dict[str, Any]]],
         meta: Optional[Dict[str, Any]] = None) -> None:
    """원본 등급을 JSON 으로 저장한다. 최종 0/1 은 담지 않는다."""
    payload = {
        "schema": SCHEMA_VERSION,
        "meta": dict(meta or {}),
        "grades": grades,
    }
    payload["meta"].setdefault("레코드수", len(grades))
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)


def load(path: str):
    """(grades, meta) 를 돌려준다. 저장된 등급은 정수로 정규화한다."""
    with io.open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or "grades" not in payload:
        raise ValueError(f"{path}: 등급 저장 형식이 아니다")
    ver = payload.get("schema")
    if ver != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema {ver} 는 이 코드({SCHEMA_VERSION})와 다르다")

    grades: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for rid, cells in (payload.get("grades") or {}).items():
        out: Dict[str, Dict[str, Any]] = {}
        for item, cell in (cells or {}).items():
            if not isinstance(cell, dict):
                continue
            ev = cell.get("근거문구")
            out[item] = {
                GRADE_KEY: int(cell.get(GRADE_KEY, 0) or 0),
                "근거문구": ev if isinstance(ev, str) else None,
            }
        grades[rid] = out
    return grades, dict(payload.get("meta") or {})


def distribution(grades: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[int, int]:
    """등급별 칸 수. 등급이 전부 0/3 이면 이진 모드로 받은 것이다."""
    out: Dict[int, int] = {}
    for cells in grades.values():
        for cell in cells.values():
            k = int(cell.get(GRADE_KEY, 0) or 0)
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))
