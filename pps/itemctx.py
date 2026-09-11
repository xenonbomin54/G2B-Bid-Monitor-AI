# -*- coding: utf-8 -*-
"""항목별 판단 문맥 생성 — item-specific context layer.

## 왜 만들었나
현재 구조는 한 번의 그룹 프롬프트에서 여러 항목을 동시에 묻는다. dev200 FP 105건
전수 분석에서 그 구조가 만드는 세 가지 실패가 확인됐다.
  1) sibling-item confusion — v24 FP 14건 중 7건이 **다른 항목 소관 문구**(지역제한)를
     근거로 들었다. 같은 프롬프트 안에 지역 사실이 있었기 때문이다.
  2) group silence — 증거가 그룹 안에 있는데도 해당 항목에 연결하지 못한다.
  3) 무관한 법령·문맥이 판단을 방해한다 — v9 FP 14건은 모델명을 정확히 찾아낸 뒤
     그것이 조달 대상인지 부품인지 구분하지 못했다. 구분에 필요한 품명 정보가
     프롬프트 어딘가에 있었지만 모델명 옆에 없었다.

## 설계
`build_item_context(rec, item, tbl, gosi)` 가 항목 하나에 대해
  · candidate_laws     — 항목표가 지정한 조문 + 그 조문의 **원문 본문**
  · evidence           — 그 항목의 키워드가 걸린 문장 + 주변 문맥
  · negative_evidence  — 위반이 아닐 수 있음을 보여주는 문장(예외 사유·대체 허용)
를 만든다. **판정하지 않는다.** 등급은 LLM 이 낸다.

법령은 `open/data/법령패키지/법령/*.txt` 원문만 쓴다. 새 DB 도 임베딩도 없다.

## 재현성
조문 파싱과 문장 추출 모두 결정론적이다. 같은 입력이면 같은 문맥이 나온다.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

from . import gosimatch, spec as _spec
from .records import Record

# 이 모듈이 만드는 문맥의 버전. **캐시 키에 들어간다** —
# 문맥 형식을 바꾸면 이 값을 올려야 이전 캐시와 섞이지 않는다.
CONTEXT_VERSION = "ictx-1"


# --------------------------------------------------------------------------- 법령 원문

# 조문 헤더는 **줄 맨 앞의 `제N조(제목)`** 이다.
# ⚠️ 괄호를 필수로 두는 이유: 법령 파일 끝의 부칙에 개정 이력이 실려 있고
#    ("제2조의2제1항제1호 본문 중 …을 …으로 한다") 그 줄이 헤더로 오인되면
#    본조 본문을 덮어써 버린다. 실제로 제2조의2 본문이 64자로 잘렸다.
#    조 번호 바로 뒤에 '제'가 오는 것(제2조의2제1항)은 **참조**이지 헤더가 아니다.
_ART_HEAD = re.compile(r"^\s{0,4}(제\d+조(?:의\d+)?)\s*\(([^)]{1,60})\)")

_LAWDIR = os.path.join("법령패키지", "법령")

_FLAT = re.compile(r"[\s·ㆍ・.()（）]")


def _flat(s: str) -> str:
    """법령명 비교용 정규화 — 공백·중점·괄호를 지운다."""
    return _FLAT.sub("", s)


# "제6장 공동계약 운영요령" 처럼 장 단위로만 지정된 참조에서 장 제목을 뽑는다.
# 항목표는 예규를 조문이 아니라 장으로 가리키는 일이 있다(v4·v9·v21).
_CHAP = re.compile(r"제\d+장\s*([^제\d]{2,30})")

# 항목표의 조문 참조 문자열에 나오는 법령 이름 → 실제 파일명 조각.
# 참조 문자열이 "국가계약법 시행령 제21조" 처럼 약칭이라 매핑이 필요하다.
_LAW_ALIAS: List[Tuple[str, str]] = [
    ("국가계약법 시행령", "국가를 당사자로 하는 계약에 관한 법률 시행령"),
    ("국가계약법 시행규칙", "국가를 당사자로 하는 계약에 관한 법률 시행규칙"),
    ("국가계약법", "국가를 당사자로 하는 계약에 관한 법률"),
    ("지방계약법 시행령", "지방자치단체를 당사자로 하는 계약에 관한 법률 시행령"),
    ("지방계약법 시행규칙", "지방자치단체를 당사자로 하는 계약에 관한 법률 시행규칙"),
    ("지방계약법", "지방자치단체를 당사자로 하는 계약에 관한 법률"),
    ("중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령",
     "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령"),
    ("중소기업제품 구매촉진 및 판로지원에 관한 법률 시행규칙",
     "중소기업제품 구매촉진 및 판로지원에 관한 법률 시행규칙"),
    ("중소기업제품 구매촉진 및 판로지원에 관한 법률",
     "중소기업제품 구매촉진 및 판로지원에 관한 법률"),
    ("중소기업기본법 시행령", "중소기업기본법 시행령"),
    ("중소기업기본법", "중소기업기본법"),
    ("소상공인기본법", "소상공인기본법"),
    ("소프트웨어 진흥법", "소프트웨어 진흥법"),
    ("지방자치단체 입찰 및 계약 집행기준", "지방자치단체 입찰 및 계약 집행기준"),
    ("지방자치단체 입찰시 낙찰자 결정기준", "지방자치단체 입찰시 낙찰자 결정기준"),
    # v9·v19 가 참조하는 계약예규. 파일은 있는데 별칭이 없어 후보가 0개였다.
    ("정부 입찰·계약 집행기준", "(계약예규) 정부 입찰·계약 집행기준"),
    ("정부 입찰ㆍ계약 집행기준", "(계약예규) 정부 입찰·계약 집행기준"),
    ("공동계약운용요령", "(계약예규) 공동계약운용요령"),
    # 항목표는 "제6장 공동계약 운영요령", 파일명은 "공동계약운용요령" —
    # '운용'/'운영' 한 글자가 달라 정규화 비교로도 안 걸렸다(v21).
    ("공동계약 운영요령", "(계약예규) 공동계약운용요령"),
    ("공동계약", "(계약예규) 공동계약운용요령"),
    ("소프트웨어 진흥법 시행령", "소프트웨어 진흥법 시행령"),
    ("중소 소프트웨어사업자의 사업 참여 지원에 관한 지침",
     "중소 소프트웨어사업자의 사업 참여 지원에 관한 지침"),
    ("중소기업자간 경쟁제품 직접생산 확인기준", "중소기업자간 경쟁제품 직접생산 확인기준"),
    ("중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역",
     "중소기업자간 경쟁제품 및 공사용자재 직접구매 대상 품목 지정 내역"),
    ("소상공인기본법 시행령", "소상공인기본법 시행령"),
]


@dataclass
class Article:
    """법령 조문 하나."""
    law: str          # 법령 이름 (파일명)
    article: str      # "제21조" · "제2조의2"
    title: str        # 조문 제목 (괄호 안)
    text: str         # 조문 본문 (항·호·목 포함)

    def head(self) -> str:
        return f"{self.law} {self.article}" + (f"({self.title})" if self.title else "")

    def blocks(self) -> List[Tuple[str, str]]:
        """조문을 (라벨, 본문) 블록으로 쪼갠다 — 항(①②) 단위, 그 안은 호(1. 2.) 단위.

        조문 전체를 통째로 주면 관련 없는 항이 판단을 흐린다. 실제 비율:
          v9  제5조 2,292자 중 v9 근거(④5호)는 약 250자 — **무관 86%**
          v17 법령 4,556자 중 관련(제2조의2①1호 + 제2조의3①)은 약 1,500자 — 무관 67%
        """
        out: List[Tuple[str, str]] = []
        paras = re.split(r"(?=^\s*[①-⑳])", self.text, flags=re.M)
        for para in paras:
            para = para.rstrip()
            if not para.strip():
                continue
            mark = re.match(r"\s*([①-⑳])", para)
            label = mark.group(1) if mark else "본문"
            # 호가 있으면 호 단위로 더 쪼갠다 — 머리글(각 호 외의 부분)은 따로 남긴다.
            hos = re.split(r"(?=^\s{0,4}\d+\.\s)", para, flags=re.M)
            if len(hos) <= 1:
                out.append((label, para))
                continue
            out.append((label, hos[0].rstrip()))
            for ho in hos[1:]:
                n = re.match(r"\s*(\d+)\.", ho)
                out.append((f"{label}{n.group(1)}호" if n else label, ho.rstrip()))
        return out or [("본문", self.text)]

    def relevant(self, focus: Sequence[str], cap: int = 1800) -> Tuple[str, bool]:
        """`focus` 키워드가 걸린 블록만 골라 조립한다.

        반환 (텍스트, 정제했는가). **하나도 안 걸리면 원문 전체를 그대로 준다** —
        recall 우선 원칙이다. 잘못 걸러 근거를 잃는 것이 무관한 조항을 함께
        주는 것보다 나쁘다.
        조문 제목 줄은 언제나 남긴다(무슨 조문인지 알아야 판단할 수 있다).
        """
        if not focus:
            return self.text[:cap], False
        blocks = self.blocks()
        keep: List[str] = []
        for label, body in blocks:
            if any(k and k in body for k in focus):
                keep.append(body)
        if not keep:
            return self.text[:cap], False
        # 조문 제목만 남긴다. 첫 **줄** 을 쓰면 줄바꿈이 없는 조문에서 ①항 전체가
        # 딸려 온다(제5조는 첫 줄이 430자다).
        head_line = f"{self.article}({self.title})" if self.title else self.article
        parts = [head_line] if not keep[0].lstrip().startswith(self.article) else []
        parts += keep
        joined = "\n".join(parts)
        if len(joined) > cap:
            joined = joined[:cap]
        # 정제가 의미 없을 만큼 거의 다 남았으면 정제하지 않은 것으로 표시한다.
        return joined, len(joined) < len(self.text) * 0.9


class LawBook:
    """법령 원문을 조문 단위로 쪼개 들고 있는 사전.

    파일을 한 번만 읽고 캐시한다 — `build_item_context` 가 항목마다 호출되므로
    매번 파싱하면 200건 × 24항목에서 느려진다.
    """

    def __init__(self, data_dir: str):
        self.dir = os.path.join(data_dir, _LAWDIR)
        self._by_law: Dict[str, Dict[str, Article]] = {}
        self._files: Dict[str, str] = {}
        if os.path.isdir(self.dir):
            for fn in os.listdir(self.dir):
                if fn.endswith(".txt"):
                    self._files[fn[:-4]] = os.path.join(self.dir, fn)

    def _load(self, law: str) -> Dict[str, Article]:
        if law in self._by_law:
            return self._by_law[law]
        out: Dict[str, Article] = {}
        path = self._files.get(law)
        if path and os.path.exists(path):
            try:
                raw = open(path, encoding="utf-8", errors="ignore").read()
            except Exception:
                raw = ""
            cur: Optional[List[Any]] = None
            for line in raw.splitlines():
                m = _ART_HEAD.match(line)
                if m:
                    if cur and cur[0] not in out:
                        # 첫 등장만 채택한다 — 뒤쪽 부칙이 본조를 덮어쓰지 않게.
                        out[cur[0]] = Article(law, cur[0], cur[1], "\n".join(cur[2]).strip())
                    cur = [m.group(1), m.group(2) or "", [line]]
                elif cur is not None:
                    cur[2].append(line)
            if cur and cur[0] not in out:
                out[cur[0]] = Article(law, cur[0], cur[1], "\n".join(cur[2]).strip())
        self._by_law[law] = out
        return out

    def resolve(self, law_hint: str) -> Optional[str]:
        """항목표의 약칭 → 파일명. 가장 긴 별칭이 먼저 걸리도록 정렬돼 있다.

        공백·중점·괄호를 지운 뒤에도 한 번 견준다 — 항목표가 같은 예규를
        "정부 입찰·계약 집행기준" 과 "정부입찰계약집행기준" 두 표기로 쓴다(v4 vs v9).
        """
        for alias, real in _LAW_ALIAS:
            if alias in law_hint and real in self._files:
                return real
        flat = _flat(law_hint)
        for alias, real in _LAW_ALIAS:
            if _flat(alias) in flat and real in self._files:
                return real
        return None

    def get(self, law: str, article: str) -> Optional[Article]:
        return self._load(law).get(article)

    def search(self, law: str, keywords: Sequence[str], limit: int = 2) -> List[Article]:
        """조문 본문에 키워드가 가장 많이 걸리는 조문. **recall 용 보조 경로**다."""
        arts = self._load(law)
        scored: List[Tuple[int, Article]] = []
        for a in arts.values():
            n = sum(1 for k in keywords if k and k in a.text)
            if n:
                scored.append((n, a))
        scored.sort(key=lambda x: (-x[0], x[1].article))
        return [a for _, a in scored[:limit]]


_BOOK: Optional[LawBook] = None


def book(data_dir: str) -> LawBook:
    global _BOOK
    if _BOOK is None or _BOOK.dir != os.path.join(data_dir, _LAWDIR):
        _BOOK = LawBook(data_dir)
    return _BOOK


# 항목표 참조 문자열에서 (법령약칭, 조문) 쌍을 뽑는다.
# 예: "국가계약법 시행령 제21조 중소기업제품 구매촉진 및 판로지원에 관한 법률 시행령
#      제2조의2 제1항,제2조의3"
#   → [("국가계약법 시행령","제21조"), ("…판로지원…시행령","제2조의2"),
#      ("…판로지원…시행령","제2조의3")]
_REF = re.compile(r"제\d+조(?:의\d+)?")

# 조문 사이에 끼는 항·호·목 참조. **법령명으로 오인하면 안 된다.**
# ⚠️ 이걸 안 걸렀을 때 실제로 무슨 일이 났는가:
#    v17 참조 "…시행령 제2조의2 제1항,제2조의3" 에서 '제1항' 이 법령명으로 잡혀
#    제2조의3 의 법령이 "제1항"이 되고, 그 결과 **v17 의 핵심 예외 조문
#    (제2조의3 우선조달계약의 예외)을 끝까지 못 가져왔다.** v14~v18 이 모두 같았다.
_SUBREF = re.compile(r"^제\d+[항호목](?:\s*[,·]\s*제\d+[항호목])*$")


def parse_refs(ref: str) -> List[Tuple[str, str]]:
    """조문 참조 문자열 → (법령약칭, 조문) 목록.

    조문 앞에 나온 법령명이 뒤따르는 조문들에 계속 이어진다. 항·호·목만 적힌
    조각은 법령명이 아니므로 **직전 법령명을 유지**한다.
    """
    out: List[Tuple[str, str]] = []
    pos = 0
    law = ""
    for m in _REF.finditer(ref):
        head = ref[pos:m.start()].strip(" ,·、")
        if head and not _SUBREF.match(head):
            law = head            # 새 법령명이 나왔으면 갈아탄다
        art = m.group(0)
        if law:
            out.append((law, art))
        pos = m.end()
    return out


# --------------------------------------------------------------------------- 항목별 키워드
#
# 각 항목의 판단에 실제로 쓰이는 어휘. **원문에 등장하는 표기 그대로** 적는다
# (공고문은 "중·소기업"·"중소기업자"·"중・소기업" 처럼 표기가 갈린다).
# positive = 그 항목이 걸릴 만한 문구 / negative = 위반이 아닐 수 있음을 보여주는 문구

@dataclass
class ItemCue:
    positive: Tuple[str, ...] = ()
    negative: Tuple[str, ...] = ()
    # 이 항목의 조문을 원문에서 더 찾을 때 쓸 키워드(recall 보조)
    law_keys: Tuple[str, ...] = ()
    # 문자열 키워드로 못 잡는 증거를 위한 정규식.
    # ⚠️ v9 이 이것 없이는 작동하지 않는다. dev200 실측에서 v9 TP 4건 중 3건의
    #    근거가 **모델명 자체**였다("Chipset: GB10 Grace Blackwell Superchip",
    #    "DJI Matrice 4E/T 시리즈", "Agilent ICP-OES 5900"). 키워드('모델'·'제조사')는
    #    그 문장에 없어서 TP 담김률이 25%(1/4)에 머물렀다.
    pos_re: Optional[Any] = None
    neg_re: Optional[Any] = None
    # 지정 조문 **안에서** 이 항목과 관련된 항·호를 고르는 키워드.
    # 비워 두면 조문 전체를 준다(기존 동작 유지).
    law_focus: Tuple[str, ...] = ()


# --------------------------------------------------------------------------- v9 전용
#
# ⚠️ spec.py 의 `_NOISE` 는 **건드리지 않는다.** 그 패턴은 baseline 프롬프트의
#    `spec.candidate_block()` 이 쓰고 있어서, 고치면 baseline 프롬프트 지문이
#    바뀌고 dev200s 재현이 깨진다. 여기서 v9 전용 잡음 패턴을 따로 둔다.
#
# 아래 목록은 dev200 audit 에서 **실제로 오인한 것만** 넣었다. 일반 제품코드를
# 광범위하게 지우면 정상 TP(GB10 · DJI Matrice 4E/T)의 recall 이 함께 죽는다.
_V9_NOISE = re.compile(
    # 용지·문서 규격 — "A4 규격, 세로방향" (PPS-DEV-101 · 140)
    r"\bA[0-9]\b\s*(규격|크기|용지|사이즈)?|한글\s*\(?HWP|\bPPT\b|\bHWP\b"
    # 목차·차시 번호 — "New 1-1-1 나에게 맞는 AI툴" (PPS-DEV-136)
    r"|(?:New|NEW)\s+\d+[-–]\d+[-–]\d+|^\s*\d+[-–]\d+[-–]\d+\s"
    # 위성항법 신호명 — "GPS(L1C/A, L2C, L5)", "BeiDou : B1, B2, B3" (PPS-DEV-088)
    r"|\bL[1-5][A-Z]?(?:/[A-Z])?\b|\bE[1-9][a-b]?\b|\bB[1-3]\b"
    r"|GLONASS|BeiDou|Galileo|SBAS|QZSS"
    # 통신 규격 버전 — "RTCM 2.1", "CMRx"
    r"|RTCM\s*\d|CMRx?\+?|NMEA"
)

# 조달 대상 품명을 meta 에서 뽑는다.
# `meta['세부품명번호목록']` 형식: "GPS[5216151801], 거리측정기[4111161302], 드론[2513189901]"
# 이 필드가 조달 대상 identity 의 **가장 정확한 출처**다(코드에서 확인했다).
_PUMNAME = re.compile(r"([^,\[\]]{2,40})\[(\d{6,12})\]")


def procurement_target(rec: Record) -> Dict[str, Any]:
    """이 공고가 **무엇을 사는가**. 판정하지 않고 사실만 모은다."""
    raw = str((rec.meta or {}).get("세부품명번호목록") or "")
    pairs = [(n.strip(), c) for n, c in _PUMNAME.findall(raw)]
    names = [n for n, _ in pairs]
    if not names:
        # 물품이 아니면(용역) 세부품명이 없다 — 그때는 공고명·용역명을 쓴다.
        try:
            names = [n for n in gosimatch.procurement_names(rec, limit=4)
                     if n and not n.startswith("[")]
        except Exception:
            names = []
    return {"품명": names, "품명번호": pairs, "업무구분": rec.업무구분 or ""}


# v9 용 모델명 패턴 — spec.py 의 검증된 정규식을 조합해 쓴다.
# 새 패턴을 발명하지 않는 이유: spec.candidate_block 이 이 패턴으로 dev200 에서
# 모델명 후보를 뽑아 왔고, 그 결과가 이미 프롬프트에 쓰이고 있다.
_MODEL_RE = re.compile(_spec._LABEL.pattern + "|" + _spec._CODE.pattern)

# 규격서·과업지시서를 **먼저** 훑어야 하는 항목.
# 판정 대상이 참가자격이 아니라 조달 물품의 규격 자체인 항목들이다.
SPEC_ITEMS = frozenset({"v9", "v12", "v19"})

_규모 = ("중소기업", "중·소기업", "중・소기업", "중소기업자", "소기업", "소상공인",
       "중기업", "확인서")
_예외 = ("2인 미만", "3인 이하", "유찰", "적격자가 없", "제2조의3", "예외",
       "비영리법인", "수의계약", "지명경쟁")

CUES: Dict[str, ItemCue] = {
    "v1": ItemCue(("대학", "산학협력단", "협회", "인증기관", "연구기관", "소속"),
                  ("업종", "면허", "등록"),
                  ("제한경쟁", "참가자격"),
                  law_focus=("특정한 명칭", "자격을 제한", "참가자격", "등록")),
    "v2": ItemCue(("실적", "납품실적", "수행실적", "준공실적", "이행실적"),
                  ("고시금액", "제한하지"), ("실적", "제한경쟁"),
                  law_focus=("실적", "고시금액 미만", "실적으로 경쟁참가자의 자격을 제한")),
    "v3": ItemCue(("실적", "배수", "이상인 실적", "준공금액", "계약금액"),
                  (), ("실적", "배수"),
                  law_focus=("실적", "1배", "3분의 1배", "규모(양)")),
    "v4": ItemCue(("국가기관", "공공기관", "지방자치단체", "발주", "납품한", "실적"),
                  (), ("실적", "제한"),
                  law_focus=("특정기관이 발주", "준공실적", "특정한 명칭의 실적", "민간실적")),
    "v5": ItemCue(("소재", "본점", "영업소", "관할구역", "지역제한"),
                  ("고시금액",), ("지역", "제한경쟁"),
                  law_focus=("지역", "관할구역", "시ㆍ도", "소재")),
    "v6": ItemCue(("시·군·구", "시군구", "관할구역", "소재", "본점"),
                  (), ("지역", "제한"),
                  law_focus=("관할구역", "시ㆍ군ㆍ자치구", "시ㆍ도")),
    "v7": ItemCue(("또는", "인접", "관할구역", "소재", "본점", "지역제한"),
                  ("예외",), ("지역", "제한"),
                  law_focus=("관할구역", "시ㆍ도", "인접")),
    "v8": ItemCue(("실적", "지역", "소재", "본점", "관할구역"),
                  (), ("중복", "제한"),
                  law_focus=("중복적으로 제한", "중복하여 제한")),
    # v9 — 규격·모델명. 조달 대상인지 부품·도구인지 가르는 것이 핵심이므로
    #      품명·규격서 어휘와 대체 허용 문구를 함께 모은다.
    # v9 의 pos_re·neg_re 는 spec.py 에 이미 검증된 패턴을 그대로 쓴다
    #  (_LABEL: 제조사·모델명·Chipset·CPU 표기 / _CODE: 영문 제품코드 GB10·RTX4090
    #   / _EQUIV: 동등이상 문구 / _NOISE: 법령 참조·날짜 등 잡음)
    "v9": ItemCue(("모델", "모델명", "제조사", "규격", "품명", "제품명", "사양",
                   "브랜드", "메이커"),
                  ("동등 이상", "동등한", "동급", "이상의 제품", "동등품", "호환",
                   "기존 장비", "참고", "예시", "권장"),
                  ("규격", "특정"),
                  pos_re=_MODEL_RE, neg_re=_spec._EQUIV,
                  law_focus=("특정상표", "모델을 지정", "특정규격", "동등이상", "납품을 거부")),
    "v10": ItemCue(("직접생산", "직생", "직접생산확인증명서"),
                   (), ("직접생산", "경쟁제품"),
                  law_focus=("직접생산", "확인", "증명")),
    "v11": ItemCue(("중소기업자", "중소기업", "제한경쟁"),
                   (), ("경쟁제품", "참여자격"),
                  law_focus=("중소기업자", "경쟁제품", "참여자격")),
    "v12": ItemCue(("직접생산", "직생", "직접생산확인증명서"),
                   ("경쟁제품",), ("직접생산",),
                  law_focus=("직접생산", "경쟁제품")),
    "v13": ItemCue(_규모, ("중기업", "중소기업 또는", "중소기업자 또는"),
                   ("경쟁제품", "제한경쟁"),
                  law_focus=("소기업", "소상공인", "경쟁제품", "중소기업자")),
    "v14": ItemCue(_규모, _예외, ("우선조달", "제한경쟁"),
                  law_focus=("중소기업자", "고시하는 금액", "우선조달계약", "제한경쟁입찰")),
    "v15": ItemCue(_규모, _예외, ("우선조달", "제한경쟁"),
                  law_focus=("소기업", "소상공인", "1억원", "중소기업자", "우선조달계약")),
    "v16": ItemCue(_규모, _예외, ("우선조달", "예외"),
                  law_focus=("중소기업자", "1억원", "우선조달계약", "체결하지 않을 수 있다", "사유를 입찰공고문")),
    # v17 — 정밀도 문제. 예외 조건(제2조의2 제1항 제1호 단서 가·나목, 제2조의3)을
    #       빠뜨리지 않는 것이 핵심이다.
    "v17": ItemCue(_규모, _예외, ("우선조달", "1억원", "예외"),
                  law_focus=("1억원", "소기업", "소상공인", "중소기업자", "제한경쟁입찰", "3인 이하", "2인 미만", "유찰", "사유를 입찰공고문")),
    "v18": ItemCue(_규모, _예외, ("우선조달", "예외"),
                  law_focus=("1억원", "소기업", "소상공인", "우선조달계약", "체결하지 않을 수 있다", "사유를 입찰공고문")),
    "v19": ItemCue(("확약서", "협약서", "공급확약", "기술지원", "제출"),
                   ("계약 체결", "계약체결 시", "낙찰 후"), ("확약서",),
                  law_focus=("확약서", "제출", "증명서")),
    "v20": ItemCue(("소프트웨어", "SW", "대기업", "참여제한", "하한"),
                   (), ("대기업", "참여제한"),
                  law_focus=("대기업", "참여", "하한", "소프트웨어")),
    "v21": ItemCue(("공동수급", "지분", "구성원", "지분율"),
                   (), ("공동수급", "지분"),
                  law_focus=("지분", "구성원", "공동수급체")),
    "v22": ItemCue(("현장설명회", "사업설명회", "설명회", "참석"),
                   ("의무가 아", "참고",), ("설명회",),
                  law_focus=("설명회", "참가자격", "협상")),
    "v23": ItemCue(("현장설명회", "사업설명회", "설명회", "마감", "제출기한"),
                   (), ("설명회", "공고기간"),
                  law_focus=("공고", "기간", "설명회")),
    "v24": ItemCue(("추정가격", "배정예산", "사업예산", "기초금액", "계약방법",
                    "낙찰방법", "입찰방법", "지역제한", "업종제한"),
                   (), ("공고",)),
}


# --------------------------------------------------------------------------- 문장 추출

_SENT_SPLIT = re.compile(r"(?<=[.!?。])\s+|\n+")


def sentences(text: str) -> List[str]:
    """문장 단위 분해. 공고문은 줄바꿈이 문장 경계 역할을 하는 경우가 많다."""
    out = []
    for chunk in _SENT_SPLIT.split(text):
        s = chunk.strip()
        if len(s) >= 6:
            out.append(s)
    return out


@dataclass
class Ev:
    text: str
    source: str      # 문서 유형
    location: str    # "문장 #12" 처럼 위치 표시
    window: str = ""  # 주변 문맥 (앞뒤 문장)


def _tier_v9(sent: str, names: Sequence[str]) -> Tuple[int, int]:
    """v9 증거의 우선순위 티어. 낮을수록 먼저 뽑힌다.

    dev200 audit 에서 `_CODE` 매치 **개수**로 정렬했더니 PPS-DEV-088 에서
    위성 신호명이 촘촘한 4줄("GPS(L1C/A, L2C, L5)", "BeiDou : B1, B2, B3")이
    증거 6칸을 독식하고, 정작 판정 근거인 "CPU : Intel Core i7-1185G7" 이
    밀려났다. 개수가 아니라 **종류**로 나눈다.

      A(0) 조달 품명과 같은 줄에 제품코드/모델 표기가 있다 → 조달 대상 특정 가능성
      B(1) 모델명·제조사 표기(_LABEL)가 있다
      C(2) 제품코드만 있다 (일반 규격)
      D(3) 문서 형식·표준 신호·목차 번호 (잡음)

    반환값은 (티어, -고유코드수) — 같은 티어 안에서는 고유 코드가 많은 쪽을 먼저.
    """
    noisy = bool(_V9_NOISE.search(sent))
    codes = set(_spec._CODE.findall(sent))
    labeled = bool(_spec._LABEL.search(sent))
    near = any(n and len(n) >= 2 and n in sent for n in names)
    if noisy and not (labeled or near):
        return (3, 0)                      # D — 잡음뿐
    if near and (codes or labeled):
        return (0, -len(codes))            # A — 품명 + 모델 표기
    if labeled:
        return (1, -len(codes))            # B — 모델명·제조사 표기
    if codes:
        return (2, -len(codes))            # C — 코드만
    return (3, 0)


def _pick_v9(sents: List[str], names: Sequence[str], limit: int,
             source: str, window: int = 1) -> List[Ev]:
    """v9 증거 선택 — 티어 우선, 원문 순서 유지."""
    scored: List[Tuple[Tuple[int, int], int]] = []
    for i, sent in enumerate(sents):
        t = _tier_v9(sent, names)
        if t[0] >= 3:
            continue                       # D 는 아예 넣지 않는다
        scored.append((t, i))
    if not scored:
        return []
    scored.sort(key=lambda x: (x[0], x[1]))
    chosen = sorted({i for _, i in scored[:limit]})
    out: List[Ev] = []
    for i in chosen:
        lo, hi = max(0, i - window), min(len(sents), i + window + 1)
        ctx = " ".join(sents[j] for j in range(lo, hi) if j != i)
        tier = "ABCD"[_tier_v9(sents[i], names)[0]]
        out.append(Ev(sents[i], f"{source}·{tier}", f"문장 #{i + 1}", ctx[:400]))
    return out


def _pick(sents: List[str], keys: Sequence[str], limit: int,
          source: str, window: int = 1, rx: Any = None,
          drop_rx: Any = None) -> List[Ev]:
    """키워드가 걸린 문장을 앞뒤 문맥과 함께 고른다.

    걸린 키워드 수가 많은 문장을 먼저 고르되, **원문 순서를 유지**해 돌려준다 —
    순서를 흩뜨리면 LLM 이 문서 흐름을 잃는다.
    """
    hits: List[Tuple[int, int]] = []
    for i, s in enumerate(sents):
        if drop_rx is not None and drop_rx.search(s):
            continue                      # 법령 참조·날짜 등 잡음 줄은 버린다
        n = sum(1 for k in keys if k and k in s)
        if rx is not None:
            n += len(rx.findall(s))       # 정규식 매치 수도 점수에 더한다
        if n:
            hits.append((n, i))
    if not hits:
        return []
    hits.sort(key=lambda x: (-x[0], x[1]))
    chosen = sorted({i for _, i in hits[:limit]})
    out: List[Ev] = []
    for i in chosen:
        lo, hi = max(0, i - window), min(len(sents), i + window + 1)
        ctx = " ".join(sents[j] for j in range(lo, hi) if j != i)
        out.append(Ev(sents[i], source, f"문장 #{i + 1}", ctx[:400]))
    return out


# --------------------------------------------------------------------------- 문맥 조립

@dataclass
class ItemContext:
    item_id: str
    item_name: str
    candidate_laws: List[Dict[str, str]] = field(default_factory=list)
    evidence: List[Ev] = field(default_factory=list)
    negative_evidence: List[Ev] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    target: Optional[Dict[str, Any]] = None   # 조달 대상(품명·업무구분)
    version: str = CONTEXT_VERSION

    def to_json(self) -> Dict[str, Any]:
        def ev(e: Ev) -> Dict[str, str]:
            return {"text": e.text, "source": e.source,
                    "location": e.location, "window": e.window}
        return {
            "item_id": self.item_id,
            "item_name": self.item_name,
            "prompt_version": self.version,
            "target": self.target,
            "candidate_laws": self.candidate_laws,
            "evidence": [ev(e) for e in self.evidence],
            "negative_evidence": [ev(e) for e in self.negative_evidence],
            "facts": self.facts,
        }

    def render(self) -> str:
        """프롬프트에 넣을 항목 문맥 블록.

        비어 있는 칸도 **'없음'이라고 명시**한다. 빈 칸으로 두면 모델이
        '문서를 못 봤다'고 여겨 그룹 전체를 다시 훑는다(group silence 의 원인).
        """
        L: List[str] = [f"■ {self.item_id} {self.item_name}"]

        if self.target:
            # **조달 대상을 모델명 옆에 붙여 준다.** v9 FP 14건의 원인이
            # "모델명은 정확히 찾았으나 그것이 조달 대상인지 몰랐다"였고,
            # 품명 정보가 프롬프트 어딘가에 있어도 모델명 옆에 없었다.
            nm = self.target.get("품명") or []
            L.append("  [조달 대상 — 이 공고가 실제로 사는 것]")
            L.append(f"   · 품명: {', '.join(nm) if nm else '(등록된 세부품명 없음)'}")
            L.append(f"   · 업무구분: {self.target.get('업무구분') or '미상'}")
            L.append("   · 아래 증거의 모델명이 **이 품명을 특정하는 것**인지, "
                     "아니면 부품·구성품·기존 장비·과업 수행 도구인지 구분하라.")

        L.append("  [관련 법령 후보]")
        if self.candidate_laws:
            for c in self.candidate_laws:
                L.append(f"   · {c['head']}")
                if c.get("reason"):
                    L.append(f"     (관련 이유: {c['reason']})")
                body = (c.get("text") or "").strip()
                if body:
                    L.append("     " + body.replace("\n", "\n     "))
        else:
            L.append("   · 후보를 찾지 못했다 — 아래 판정 지침만으로 판단하라")

        if self.facts:
            L.append("  [계산된 사실]")
            for f in self.facts:
                L.append(f"   · {f}")

        L.append("  [핵심 증거]")
        if self.evidence:
            for e in self.evidence:
                L.append(f"   {e.location} ({e.source}) {e.text}")
                if e.window:
                    L.append(f"     …주변: {e.window}")
        else:
            L.append("   · **이 항목의 키워드가 걸린 문장을 찾지 못했다.**")
            L.append("     찾지 못한 것이 곧 '위반이 아니다'라는 뜻은 아니다 — "
                     "부재가 위반인 항목도 있다.")

        L.append("  [반대 증거 — 위반이 아닐 수 있는 근거]")
        if self.negative_evidence:
            for e in self.negative_evidence:
                L.append(f"   {e.location} ({e.source}) {e.text}")
        else:
            L.append("   · 없음")
        return "\n".join(L)


def _has_focus(ctx: "ItemContext", focus: Sequence[str]) -> bool:
    """후보 조문 본문에 항목 초점 어휘가 실제로 들어 있는가."""
    if not focus:
        return bool(ctx.candidate_laws)
    blob = "\n".join(c["text"] for c in ctx.candidate_laws)
    return any(k and k in blob for k in focus)


def build_item_context(
    rec: Record,
    item: str,
    tbl: Dict[str, Dict[str, Any]],
    data_dir: str,
    gosi: Any = None,
    max_laws: int = 3,
    max_ev: int = 6,
    max_neg: int = 4,
) -> ItemContext:
    """항목 하나에 대한 독립 문맥. **판정하지 않는다.**"""
    it = tbl.get(item, {})
    ctx = ItemContext(item_id=item, item_name=it.get("항목명", item))
    cue = CUES.get(item, ItemCue())
    bk = book(data_dir)

    # ---- 법령 후보 (Phase 2) --------------------------------------------
    # 이 공고의 계약법에 맞는 쪽을 **먼저** 보고, 부족하면 반대쪽도 본다.
    # ⚠️ 반대쪽까지 보는 이유: v4·v9·v21 의 지방계약법 참조는 조문이 아니라
    #    "제1장"·"제6장" 같은 **장(章) 단위**여서 조문 정규식에 걸리지 않는다.
    #    국가계약법 쪽에는 같은 내용이 조문으로 지정돼 있다(v9 → 계약예규 제5조).
    #    recall 을 정확도보다 앞세운다는 원칙에 따라 양쪽을 다 시도한다.
    primary = "지방계약법" if rec.적용계약법 and "지방" in rec.적용계약법 else "국가계약법"
    other = "국가계약법" if primary == "지방계약법" else "지방계약법"
    seen = set()

    def add(real: str, art: str, why: str, cap: int) -> None:
        if (real, art) in seen or len(ctx.candidate_laws) >= max_laws:
            return
        a = bk.get(real, art)
        if a is None:
            return
        seen.add((real, art))
        # raw reference(법령명·조문번호)는 그대로 보존하고, **LLM 에 넣는 본문만**
        # 항·호 단위로 정제한다. 항목표도 법령 원문도 건드리지 않는다.
        body, refined = a.relevant(cue.law_focus, cap)
        ctx.candidate_laws.append(
            {"head": a.head(), "source": real, "article": art,
             "text": body, "reason": why,
             "raw_len": len(a.text), "refined": refined})

    for field_name, why in (
            (primary, f"항목표가 이 항목의 근거 조문으로 지정({primary})"),
            (other, f"{primary} 쪽 참조가 조문 단위가 아니어서 {other} 참조로 보완")):
        ref = str(it.get(field_name) or "")
        for law_hint, art in parse_refs(ref):
            real = bk.resolve(law_hint)
            if real:
                add(real, art, why, 1800)
        if len(ctx.candidate_laws) >= max_laws:
            break

    # recall 보조 — **조문 단위 참조가 하나도 없을 때만** 쓴다.
    # 장 단위 참조("… 집행기준 제6장 공동계약 운영요령")는 조문 번호가 없으므로
    # **그 장의 제목을 검색 키로** 쓴다.
    # ⚠️ 항목 키워드(CUES.law_keys)로 훑는 방식을 먼저 썼다가 v21 에서
    #    "제10조의2(비용의 분담)" 같은 무관한 조문이 1위로 올라왔다. 무관한 법령을
    #    끼워 coverage 를 부풀리는 것은 판단을 방해하므로(이번 실험의 가설 자체가
    #    'irrelevant context 가 판단을 흐린다'이다) **맞는 것이 없으면 비워 둔다.**
    if not _has_focus(ctx, cue.law_focus):
        for field_name in (primary, other):
            hint = str(it.get(field_name) or "")
            chap = _CHAP.findall(hint)
            if not chap:
                continue
            keys = tuple(w for t in chap
                         for w in re.split(r"\s+", t.strip()) if len(w) >= 2)
            # 장 제목만으로는 그 장 안의 **어느 조문**인지 못 고른다.
            # v21 에서 "공동계약 운영요령" 으로 찾으니 제목에 '공동계약'이 든
            # 제2조의2(공동계약의 유형)가 1위로 왔고, 정작 최소지분율이 있는
            # 제9조(공동수급체의 구성)를 놓쳤다. 항목의 law_focus 를 함께 쓴다.
            keys = keys + tuple(cue.law_focus)
            if not keys:
                continue
            # 별칭 후보를 **장 제목과 맞는 것부터** 본다.
            # v21 참조에는 "지방자치단체 입찰 및 계약 집행기준"과
            # "공동계약 운영요령"이 함께 있는데, _LAW_ALIAS 순서대로 보면
            # 앞의 집행기준이 먼저 걸려 제14~16조(운영위원회)가 3칸을 채우고
            # 정작 장이 가리키는 공동계약운용요령에 도달하지 못한다.
            chap_txt = " ".join(chap)
            cands: List[Tuple[int, str, str]] = []
            for alias, real in _LAW_ALIAS:
                if not (alias in hint or _flat(alias) in _flat(hint)):
                    continue
                if real not in bk._files:
                    continue
                # 장 제목에 별칭이 들어 있으면 그 법령이 이 장의 본체다
                pri = 0 if (_flat(alias) in _flat(chap_txt)) else 1
                cands.append((pri, alias, real))
            cands.sort(key=lambda x: x[0])
            for _pri, alias, real in cands:
                # 조문 **제목**에 장 제목 키워드가 걸린 것만 받는다.
                # 본문 매칭만 보면 v21 에서 "제14조(운영위원회)" 처럼 '운영'
                # 한 조각이 겹친 무관한 조문이 올라왔다.
                # 조문 **제목** 이나 **본문** 에 항목 초점 어휘가 걸린 것.
                # 제목만 보면 v21 처럼 내용이 맞는 조문을 놓친다.
                cand = bk.search(real, keys, limit=max_laws * 8)
                # 장 단위 참조에서는 **초점 어휘가 본문에 실제로 있는 조문만** 받는다.
                # 제목 매칭만 허용하면 v21 에서 제14~16조(운영위원회)가 3칸을 채워
                # 정작 최소지분율이 있는 공동계약운용요령 제9조에 도달하지 못했다.
                if cue.law_focus:
                    hits = [a for a in cand
                            if any(k in a.text for k in cue.law_focus)]
                else:
                    hits = [a for a in cand if any(k in a.title for k in keys)]
                for a in hits[:max_laws]:
                    add(real, a.article,
                        f"{field_name} 참조가 장 단위여서 그 장 제목·항목 초점"
                        f"('{' '.join(keys[:4])}')으로 찾은 조문", 1200)
                # ⚠️ 후보가 생겼다고 바로 멈추면 안 된다. v21 참조에는
                #    "지방자치단체 입찰 및 계약 집행기준"과 "공동계약 운영요령"이
                #    함께 들어 있어, 앞 별칭이 먼저 걸리면 제14~16조(운영위원회)로
                #    채워지고 정작 최소지분율이 있는 공동계약운용요령 제9조에
                #    도달하지 못한다. **초점 어휘가 실제로 든 조문**을 찾을 때까지 간다.
                if _has_focus(ctx, cue.law_focus):
                    break
            if _has_focus(ctx, cue.law_focus):
                break

    # 항목표 비고는 조문이 아니지만 판정 조건이므로 사실로 넘긴다.
    if it.get("비고"):
        ctx.facts.append(f"항목표 비고: {it['비고']}")

    # ---- 증거 (Phase 3) --------------------------------------------------
    # 문서 순서가 곧 증거 예산의 배분이다. 앞 문서에서 max_ev 를 다 쓰면
    # 뒤 문서는 한 문장도 못 들어온다.
    # ⚠️ 실측한 실패: v9(과업지시서 특정 모델명)에서 공고문을 먼저 훑자
    #    PPS-DEV-088 의 증거 6칸이 공고문 **품명 나열표**로 다 채워지고,
    #    정작 판정 대상인 규격서의 "CPU : Intel Core i7-1185G7" 이 들어오지 못했다.
    #    → 항목이 규격·모델을 보는 것이면 규격서·과업지시서를 **먼저** 본다.
    docs: Dict[str, str] = {"공고문": rec.notice_text}
    for d in (rec.docs or []):
        t = d.get("type") or "첨부"
        body = d.get("text") or ""
        if t != "공고문" and body:
            docs[t] = docs.get(t, "") + ("\n" if t in docs else "") + body

    _SPEC_FIRST = ("규격서", "과업지시서", "제안요청서", "시방서", "물품규격서")
    if item in SPEC_ITEMS:
        order = ([t for t in _SPEC_FIRST if t in docs]
                 + [t for t in docs if t not in _SPEC_FIRST])
    else:
        order = ["공고문"] + [t for t in docs if t != "공고문"]
    pools: List[Tuple[str, str]] = [(t, docs[t]) for t in order if docs.get(t)]

    if item == "v9":
        # v9 은 조달 대상 identity 가 판정의 축이므로 문맥에 함께 싣는다.
        ctx.target = procurement_target(rec)
        names = ctx.target["품명"]
        for src, body in pools:
            if len(ctx.evidence) >= max_ev:
                break
            ss = sentences(body)
            ctx.evidence.extend(
                _pick_v9(ss, names, max_ev - len(ctx.evidence), src))
    else:
        for src, body in pools:
            if len(ctx.evidence) >= max_ev:
                break
            ss = sentences(body)
            room = max_ev - len(ctx.evidence)
            ctx.evidence.extend(_pick(ss, cue.positive, room, src,
                                      rx=cue.pos_re, drop_rx=None))
    for src, body in pools:
        if len(ctx.negative_evidence) >= max_neg:
            break
        ss = sentences(body)
        room = max_neg - len(ctx.negative_evidence)
        ctx.negative_evidence.extend(
            _pick(ss, cue.negative, room, src, window=0, rx=cue.neg_re))
    return ctx


# --------------------------------------------------------------------------- 그룹 렌더링
#
# grouped call 을 유지하면서 **항목 경계를 명확히** 하는 것이 목적이다.
# 그룹 전체의 법령·증거를 하나로 합치지 않는다 — 합치면 sibling-item confusion 이
# 그대로 남는다(v24 FP 14건 중 7건이 다른 항목 소관 문구를 근거로 들었다).
#
# 예산: G2 는 게이팅 후 평균 5.1항목이고 정제 후 법령이 평균 11,897자 · 최대 14,317자다.
# baseline 프롬프트가 이미 최대 5,000 토큰을 쓰고 예산은 15,184 토큰이므로
# 항목 문맥을 그대로 다 실으면 넘친다. 그래서 **문자 예산**을 둔다.
# ⚠️ 예산을 맞출 때 긴 조문을 중간에서 자르지 않는다. v9 에서 근거가 조문 뒤쪽
#    67% 지점에 있음을 확인했다. 이미 항·호 단위로 정제된 **블록 단위로 통째로**
#    넣거나 뺀다.
GROUP_LAW_BUDGET = 7000      # 그룹 전체 법령 본문 문자 예산
GROUP_EV_BUDGET = 3500       # 그룹 전체 증거 문자 예산


def _focus_score(text: str, focus: Sequence[str]) -> int:
    return sum(1 for k in focus if k and k in text)


def render_group(ctxs: Sequence[ItemContext],
                 law_budget: int = GROUP_LAW_BUDGET,
                 ev_budget: int = GROUP_EV_BUDGET) -> str:
    """항목별 문맥을 **경계를 세워** 이어 붙인다.

    배분 규칙 (deterministic)
      1. 조문을 항목별 초점 매칭 강도로 정렬한다 — v17 은 제2조의2(1억원·소기업·
         단서 가·나목이 모두 걸린다)가 1순위가 되어야 한다. 항목표 참조 순서가
         아니라 **내용 관련도** 순이다.
      2. 라운드로빈으로 각 항목의 1순위부터 채운다. 한 항목이 예산을 독식하지 못한다.
      3. 같은 조문이 여러 항목에 걸리면 본문은 **한 번만** 싣고 이후로는
         "(ITEM vNN 에 실린 조문과 같다)"로 가리킨다. v14~v18 이 제20조·제2조의2를
         공유해 그룹당 평균 2,075자가 중복된다.
      4. 예산이 모자라면 그 조문을 **통째로 뺀다**. 잘라서 넣지 않는다.
    """
    order = [c.item_id for c in ctxs]
    ranked: Dict[str, List[Dict[str, str]]] = {}
    for c in ctxs:
        foc = CUES.get(c.item_id, ItemCue()).law_focus
        ranked[c.item_id] = sorted(
            c.candidate_laws,
            key=lambda x: (-_focus_score(x["text"], foc), x["article"]))

    placed: Dict[Tuple[str, str], str] = {}     # (법령, 조문) → 처음 실은 항목
    chosen: Dict[str, List[Tuple[Dict[str, str], Optional[str]]]] = {i: [] for i in order}
    used = 0
    for rnd in range(3):                        # 1순위 → 2순위 → 3순위
        for item in order:
            laws = ranked.get(item) or []
            if rnd >= len(laws):
                continue
            law = laws[rnd]
            key = (law["source"], law["article"])
            if key in placed:
                chosen[item].append((law, placed[key]))   # 참조만
                continue
            if used + len(law["text"]) > law_budget and chosen[item]:
                continue                        # 이미 1개는 받았으니 건너뛴다
            if used + len(law["text"]) > law_budget:
                continue
            placed[key] = item
            chosen[item].append((law, None))
            used += len(law["text"])

    ev_used = 0
    out: List[str] = []
    for idx, c in enumerate(ctxs, 1):
        out.append(f"\n{'━' * 58}")
        out.append(f"[ITEM {idx}] {c.item_id} — {c.item_name}")
        out.append(f"{'━' * 58}")
        out.append(f"※ 이 블록의 정보는 **{c.item_id} 판정에만** 쓴다. "
                   f"다른 ITEM 의 법령·증거를 끌어오지 마라.")

        if c.target:
            nm = c.target.get("품명") or []
            out.append(f"  · 조달 대상 품명: {', '.join(nm) if nm else '(등록된 세부품명 없음)'}"
                       f" / 업무구분: {c.target.get('업무구분') or '미상'}")
        if c.facts:
            for f in c.facts:
                out.append(f"  · {f}")

        out.append("  【관련 법령】")
        got = chosen.get(c.item_id) or []
        if not got:
            out.append("   · 지정된 조문이 없다(조문 없는 대조형 항목이거나 "
                       "참조를 찾지 못했다). 판정 지침만으로 판단하라.")
        for law, ref in got:
            if ref:
                out.append(f"   · {law['head']} → ITEM 의 {ref} 에 실린 조문과 같다.")
                continue
            out.append(f"   · {law['head']}")
            out.append("     " + law["text"].strip().replace("\n", "\n     "))

        out.append("  【공고문에서 찾은 증거】")
        if not c.evidence:
            out.append("   · 이 항목의 키워드가 걸린 문장을 찾지 못했다. "
                       "**부재가 곧 위반인 항목도 있으니** 판정 지침을 보라.")
        for e in c.evidence:
            line = f"   · [{e.source}] {e.text}"
            if ev_used + len(line) > ev_budget:
                out.append("   · (증거 예산 초과로 이하 생략)")
                break
            ev_used += len(line)
            out.append(line)

        if c.negative_evidence:
            out.append("  【반대 증거 — 위반이 아닐 수 있는 근거】")
            for e in c.negative_evidence[:3]:
                out.append(f"   · [{e.source}] {e.text[:200]}")
    return "\n".join(out)


def build_group_context(rec: Record, items: Sequence[str],
                        tbl: Dict[str, Dict[str, Any]], data_dir: str,
                        gosi: Any = None) -> Tuple[str, List[ItemContext]]:
    """그룹 한 번 호출에 들어갈 항목별 문맥 전체."""
    ctxs = [build_item_context(rec, i, tbl, data_dir, gosi=gosi) for i in items]
    return render_group(ctxs), ctxs
