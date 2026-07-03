"""
schema.py — 모든 자료의 공통 레코드 구조.

핵심 원칙: 출처 태그 없는 데이터는 시스템에 들어올 수 없다.
레이어4의 해설이 "출처를 밝혀서" 나오려면, 애초에 모든 원본이
어디서 왔는지를 들고 있어야 한다.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import hashlib


# 허용되는 값들 (오타·혼란 방지용 화이트리스트)
LAYERS = {"L1_pattern", "L2_corpus", "L3_trend", "L4_generate"}
LEVELS = {"초등", "중등", "특수", "공통"}   # 급별 (기출 교차분석용)
SUBJECTS = {
    "총론", "창의적체험활동", "통합교과",
    "국어", "영어", "수학", "사회", "과학",
    "미술", "음악", "체육", "실과", "도덕",
}


@dataclass
class Record:
    """
    자료 한 조각(문장/문단/문항 하나)의 최소 단위.

    필수 태그(없으면 거부):
      - text   : 실제 내용
      - layer  : 어느 레이어의 자료인가
      - subject: 어느 과목인가
      - source : 출처(파일명/URL/책이름+페이지 등) — 절대 비울 수 없음
    선택 태그:
      - year   : 기출이면 출제연도 (L1에서 시계열 분석에 필수)
      - level  : 초/중/특/공통 (기출 교차분석)
      - code   : 성취기준 코드 [4국01-01] 등 (있으면 매핑 정확도↑)
      - qtype  : 발문 형식(서술형/사례형/객관식 등) — 나중 확장용
    """
    text: str
    layer: str
    subject: str
    source: str                      # ← 출처. 무조건 채워야 함.
    year: Optional[int] = None
    level: Optional[str] = None
    code: Optional[str] = None
    qtype: Optional[str] = None
    rec_id: str = field(default="")

    def __post_init__(self):
        # --- 출처 태그 강제 ---
        if not self.text or not self.text.strip():
            raise ValueError("Record.text 가 비어있음")
        if not self.source or not self.source.strip():
            raise ValueError(f"출처(source) 없는 데이터는 거부됨: {self.text[:30]!r}")
        if self.layer not in LAYERS:
            raise ValueError(f"알 수 없는 layer={self.layer!r} (허용:{LAYERS})")
        if self.subject not in SUBJECTS:
            raise ValueError(f"알 수 없는 subject={self.subject!r} (허용:{SUBJECTS})")
        if self.level is not None and self.level not in LEVELS:
            raise ValueError(f"알 수 없는 level={self.level!r} (허용:{LEVELS})")
        # L1(기출 패턴)은 연도가 반드시 있어야 시계열 분석 가능
        if self.layer == "L1_pattern" and self.year is None:
            raise ValueError("L1_pattern 레코드는 year(출제연도)가 필수")
        # 안정적 id (같은 내용+출처면 같은 id → 중복 방지)
        if not self.rec_id:
            h = hashlib.md5(f"{self.source}|{self.text}".encode("utf-8")).hexdigest()[:12]
            self.rec_id = h

    def to_dict(self):
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Record":
        return Record(**d)


def save_records(records: list[Record], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump([r.to_dict() for r in records], f, ensure_ascii=False, indent=2)


def load_records(path: str) -> list[Record]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [Record.from_dict(d) for d in data]


# ── pkl 저장/로드 (Record 객체 그대로 보존) ──────────────────────
import pickle as _pickle

def save_records_pkl(records, path):
    """Record 리스트를 pkl로 저장(사용자 원본 자료용)."""
    with open(path, "wb") as f:
        _pickle.dump([r.to_dict() for r in records], f)

def load_records_pkl(path):
    """pkl에서 Record 리스트 복원. 없으면 빈 리스트."""
    import os
    if not os.path.exists(path):
        return []
    with open(path, "rb") as f:
        data = _pickle.load(f)
    return [Record.from_dict(d) for d in data]
