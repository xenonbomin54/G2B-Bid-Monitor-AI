# -*- coding: utf-8 -*-
"""레코드 로딩·정규화·메타 파싱.

한글은 전부 NFC로 맞춘다. 근거문구를 원문과 대조할 때 NFD가 섞이면
`in` 검사가 조용히 실패하고, 그 결과 근거가 통째로 버려진다.
"""
from __future__ import annotations

import gzip
import io
import json
import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from . import law

ITEMS = [f"v{i}" for i in range(1, 25)]
EVID = [f"e{i}" for i in range(1, 25)]
COLUMNS = ["id"] + ITEMS + EVID

# 부재탐지 항목 — "있어야 할 기재가 없는 것"이 위반이므로 인용할 원문이 없다.
ABSENCE = frozenset(["v10", "v11", "v16", "v18", "v20"])

DOC_ORDER = ["공고문", "규격서", "과업지시서", "제안요청서", "예외공표서", "기타"]

META_FIELDS = [
    "적용계약법", "업무구분", "계약방법", "낙찰방법", "낙찰하한율",
    "배정예산금액", "입찰추정가격", "소관구분", "공동도급구성방식", "정보화사업여부",
    "세부품명번호목록", "제한지역코드목록", "지역제한여부", "면허업종제한목록", "업종제한여부",
    "조항호내용", "공고게시일자", "개찰예정일자", "긴급공고여부", "입찰방법", "조달방식",
]

EVIDENCE_MAX = 500


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s) if isinstance(s, str) else s


def _open(path: str):
    if str(path).endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8")
    return io.open(path, "r", encoding="utf-8")


# --------------------------------------------------------------------------- 레코드

@dataclass
class Record:
    """입찰공고 1건."""
    id: str
    docs: List[Dict[str, Any]]
    meta: Dict[str, Any]
    dropped_doc_counts: Dict[str, int] = field(default_factory=dict)
    input_completeness: Dict[str, Any] = field(default_factory=dict)
    raw: Dict[str, Any] = field(default_factory=dict)

    # ---- 문서 접근 -------------------------------------------------------
    def text_of(self, *types: str) -> str:
        """지정 종류 문서만 이어붙인다. 인자가 없으면 전부."""
        docs = self.docs if not types else [d for d in self.docs if d["type"] in types]
        return "\n".join(d["text"] for d in docs)

    @property
    def full_text(self) -> str:
        """근거문구 대조용 원문 (NFC)."""
        return self.text_of()

    @property
    def notice_text(self) -> str:
        return self.text_of("공고문")

    # ---- 메타 파생값 ------------------------------------------------------
    @property
    def 적용계약법(self) -> str:
        return self.meta.get("적용계약법") or ""

    @property
    def 업무구분(self) -> str:
        return self.meta.get("업무구분") or ""

    @property
    def 계약방법(self) -> str:
        return self.meta.get("계약방법") or ""

    @property
    def 낙찰방법(self) -> str:
        return self.meta.get("낙찰방법") or ""

    @property
    def 추정가격(self) -> int:
        """입찰추정가격. 없으면 배정예산금액으로 대체(부가세 포함이라 과대추정)."""
        v = self.meta.get("입찰추정가격")
        if isinstance(v, (int, float)) and v > 0:
            return int(v)
        v = self.meta.get("배정예산금액")
        return int(v) if isinstance(v, (int, float)) and v > 0 else 0

    @property
    def 배정예산(self) -> int:
        v = self.meta.get("배정예산금액")
        return int(v) if isinstance(v, (int, float)) and v > 0 else 0

    @property
    def is_지방(self) -> bool:
        return self.적용계약법 == "지방계약법"

    @property
    def is_물품(self) -> bool:
        return "물품" in self.업무구분

    @property
    def is_공사(self) -> bool:
        return "공사" in self.업무구분

    @property
    def is_협상(self) -> bool:
        return self.낙찰방법 == "협상에의한계약"

    @property
    def is_수의(self) -> bool:
        return "수의" in self.계약방법 or "수의" in self.낙찰방법

    @property
    def 판로지원밴드(self) -> str:
        """'A'(<1억) / 'B'(1억~고시금액) / 'C'(≥고시금액)."""
        return law.판로지원_밴드(self.추정가격, self.업무구분)

    @property
    def 소액수의_지방(self) -> bool:
        """지방 + 추정가격 1억 이하 물품·용역 → 2인견적 수의계약 경로.

        항목표 비고 "지방 + 소액수의 가능"(v2·v6·v7·v8)의 예외 조건.
        """
        return (
            self.is_지방
            and not self.is_공사
            and 0 < self.추정가격 <= law.소액수의_상한_지방_물품용역
        )

    @property
    def 세부품명번호(self) -> List[str]:
        """meta.세부품명번호목록에서 10자리 번호 추출. 형식: '잡지[5510150601]'."""
        s = self.meta.get("세부품명번호목록")
        return re.findall(r"\d{10}", str(s)) if s else []

    @property
    def 본문_세부품명번호(self) -> List[str]:
        """본문에 등장하는 10자리 세부품명번호."""
        return re.findall(r"\d{10}", self.full_text)


# --------------------------------------------------------------------------- 검증·로딩

def validate(rec: Any) -> None:
    if not isinstance(rec, dict):
        raise ValueError(f"레코드가 object가 아니다: {type(rec).__name__}")
    for k in ("id", "docs", "meta"):
        if k not in rec:
            raise ValueError(f"필수 키 없음: {k}")
    if not isinstance(rec["id"], str) or not rec["id"]:
        raise ValueError("id가 비어 있다")
    docs = rec["docs"]
    if not isinstance(docs, list) or not docs:
        raise ValueError(f"docs가 비어 있다 (id={rec['id']})")
    for d in docs:
        if not isinstance(d, dict) or not all(k in d for k in ("doc_id", "type", "text")):
            raise ValueError(f"docs 원소 형식 오류 (id={rec['id']})")
        if not isinstance(d["text"], str):
            raise ValueError(f"docs.text가 문자열이 아니다 (id={rec['id']})")
    if not any(d["type"] == "공고문" for d in docs):
        raise ValueError(f"공고문이 없다 (id={rec['id']})")
    if not isinstance(rec["meta"], dict):
        raise ValueError(f"meta가 object가 아니다 (id={rec['id']})")


def _to_record(raw: Dict[str, Any]) -> Record:
    docs = []
    for d in raw["docs"]:
        docs.append({
            "doc_id": d["doc_id"],
            "type": nfc(d["type"]),
            "text": nfc(d["text"]),
        })
    return Record(
        id=raw["id"],
        docs=docs,
        meta=raw["meta"],
        dropped_doc_counts=raw.get("dropped_doc_counts") or {},
        input_completeness=raw.get("input_completeness") or {},
        raw=raw,
    )


def iter_records(path: str, limit: Optional[int] = None) -> Iterator[Record]:
    n = 0
    with _open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno} JSON 파싱 실패: {e}") from e
            validate(raw)
            yield _to_record(raw)
            n += 1
            if limit and n >= limit:
                return


def load_records(path: str, limit: Optional[int] = None) -> List[Record]:
    return list(iter_records(path, limit))


# --------------------------------------------------------------------------- 항목표

def load_item_table(data_dir: str) -> Dict[str, Dict[str, Any]]:
    import os
    p = os.path.join(data_dir, "항목표.json")
    with io.open(p, encoding="utf-8") as f:
        return json.load(f)["항목"]
