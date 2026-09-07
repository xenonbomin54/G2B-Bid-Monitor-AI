# -*- coding: utf-8 -*-
"""근거문구 정합 — 모델이 낸 인용을 원문의 실제 부분문자열로 복원한다.

왜 중요한가
  2차 평가(정성평가 20%)는 "제공 원문의 부분문자열이 아닌 근거 문구는
  자동 검증 단계에서 무효 처리"한다. 베이스라인은 부분문자열이 아니면
  그냥 버린다 — 모델이 공백 하나만 다르게 옮겨도 근거가 통째로 날아간다.

  여기서는 버리기 전에 복원을 시도한다.
    1) 그대로 맞으면 그대로
    2) 공백·줄바꿈만 다르면 원문 쪽 실제 구간을 찾아 되돌린다
    3) 앞뒤 군더더기(따옴표·말줄임표·번호)만 붙었으면 벗겨낸다
    4) 그래도 안 되면 가장 긴 공통 구간으로 축소한다 (짧아도 정답 근거와
       겹치면 정성평가에서 유리하다 — "위반에 해당하는 부분만 인용"이 기준)
    5) 전부 실패하면 빈칸 (원문에 없는 근거는 무효이므로 내보내지 않는다)

리더보드 점수(v열)에는 영향이 없다. 2차 평가 재료다.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Optional

EVIDENCE_MAX = 500

_WS = re.compile(r"\s+")
# 모델이 인용 앞뒤에 흔히 붙이는 군더더기
_TRIM = ' \t\r\n"\'“”‘’「」『』<>《》()[]{}…·-–—:;,.'


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def _ws_flexible_pattern(s: str) -> re.Pattern:
    """공백 위치·개수 차이를 흡수하는 정규식. 나머지 문자는 그대로 요구한다."""
    parts = [re.escape(p) for p in _WS.split(s.strip()) if p]
    return re.compile(r"\s*".join(parts))


def _longest_common_substring(a: str, b: str, min_len: int = 12) -> Optional[str]:
    """a 안에서 b 와 겹치는 가장 긴 연속 구간. 짧으면 None.

    O(len(a)·len(b)) 를 피하려고 a 의 후보 길이를 이분 탐색한다.
    """
    if not a or not b:
        return None

    def has(n: int) -> Optional[str]:
        # b는 짧은 인용문, a는 수만 자 원문이다. 원문의 모든 부분문자열을
        # Python set으로 만들지 않고 C 구현의 문자열 검색을 사용한다.
        # b의 앞쪽 후보부터 검사하므로 동률 선택도 기존 구현과 같다.
        for i in range(len(b) - n + 1):
            candidate = b[i:i + n]
            if candidate in a:
                return candidate
        return None

    lo, hi, best = min_len, min(len(a), len(b)), None
    if hi < min_len:
        return None
    while lo <= hi:
        mid = (lo + hi) // 2
        got = has(mid)
        if got:
            best, lo = got, mid + 1
        else:
            hi = mid - 1
    return best


def align(quote: Optional[str], source: str, max_len: int = EVIDENCE_MAX) -> str:
    """모델 인용을 원문 부분문자열로 정합한다. 실패하면 빈 문자열."""
    if not quote:
        return ""
    src = _nfc(source)
    q = _nfc(quote).replace("\r", "").strip()
    if not q:
        return ""

    # 1) 그대로 일치
    if q in src:
        return _cap(q, max_len)

    # 2) 앞뒤 군더더기 제거 후 일치
    stripped = q.strip(_TRIM)
    if stripped and stripped in src:
        return _cap(stripped, max_len)

    # 3) 공백만 다른 경우 — 원문 쪽 실제 표기를 가져온다
    for cand in (q, stripped):
        if not cand:
            continue
        try:
            m = _ws_flexible_pattern(cand).search(src)
        except re.error:
            m = None
        if m:
            return _cap(m.group(0), max_len)

    # 4) 가장 긴 공통 구간으로 축소
    lcs = _longest_common_substring(src, stripped or q)
    if lcs:
        lcs = lcs.strip(_TRIM)
        if len(lcs) >= 12 and lcs in src:
            return _cap(lcs, max_len)

    return ""


def _cap(s: str, max_len: int) -> str:
    """길이 상한 + 수식 접두 방어.

    CSV 셀이 '=' '+' '@' 로 시작하면 스프레드시트에서 수식으로 해석된다.
    제출 규약이 금지하므로 벗겨낸다.
    """
    s = s[:max_len].strip()
    while s and s[0] in "=+@":
        s = s[1:].lstrip()
    return s


def clean(quote: Optional[str], source: str, is_absence: bool, is_violation: bool) -> str:
    """제출용 근거문구 최종 형태.

    · 부재탐지 항목은 인용할 원문이 없으므로 항상 빈칸
    · 위반이 아니면 빈칸
    · 그 외에는 원문 정합을 시도
    """
    if is_absence or not is_violation:
        return ""
    return align(quote, source)
