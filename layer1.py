"""
레이어1 — 기출 시계열 패턴 분석 (국어)

역할:
  역대 기출(초/중/특)을 연도 태그와 함께 L2의 고정 임베딩+SOM 위에 올려,
  '같은 개념 영역(BMU 클러스터)이 어느 해에 몰려 출제됐는가'를 본다.

패턴 정의(합의됨): '같은 개념 영역'.
  → 같은 SOM 노드(또는 인접 노드)에 배정된 기출들을 한 개념군으로 보고,
    그 개념군의 연도 분포를 확인한다.

급별 교차분석(합의됨):
  각 기출에 level(초/중/특) 태그가 있으므로,
  "중등에서 먼저 나온 개념이 몇 년 뒤 초등에 내려왔나"를 볼 수 있다.

주의: 이건 '예측기'가 아니라 '경향 분석기'다.
  어느 개념이 자주/최근 다뤄지는지를 보여줄 뿐, 올해 정답을 맞히지 않는다.
"""
from __future__ import annotations
import sys, os
import numpy as np
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "core"))
from embedding import FrozenEmbedding
from som import SOM
from korean_tokenizer import tokenize
from schema import load_records


def map_exams_to_som(exam_records, emb: FrozenEmbedding, som: SOM):
    """기출 레코드를 (학습된) 국어 SOM에 얹어 BMU를 구한다. SOM은 재학습하지 않음."""
    node_exams = defaultdict(list)   # node -> [(year, level, rec)]
    for r in exam_records:
        v = emb.embed_tokens(tokenize(r.text))
        if v is None:
            continue
        b = som.bmu_of(v)
        node_exams[b].append((r.year, r.level, r))
    return node_exams


def concept_year_report(node_exams, min_hits=2):
    """
    각 개념영역(노드)별 연도 분포 리포트.
    min_hits: 이 개념군에 이만큼 이상 기출이 있어야 '패턴'으로 본다.
    """
    report = []
    for node, items in node_exams.items():
        if len(items) < min_hits:
            continue
        years = sorted(y for y, lv, r in items if y is not None)
        levels = [lv for y, lv, r in items if lv]
        sample = items[0][2].text[:50]
        report.append({
            "node": node,
            "hit_count": len(items),
            "years": years,
            "year_span": (min(years), max(years)) if years else None,
            "levels": dict(_count(levels)),
            "sample": sample,
        })
    report.sort(key=lambda d: -d["hit_count"])
    return report


def cross_level_flow(node_exams):
    """
    급별 교차분석: 같은 개념영역에서 급별로 '처음 등장한 해'를 비교.
    중등이 초등보다 먼저면 → 하향 전파 후보.
    """
    flows = []
    for node, items in node_exams.items():
        first_by_level = {}
        for y, lv, r in items:
            if y is None or not lv:
                continue
            if lv not in first_by_level or y < first_by_level[lv]:
                first_by_level[lv] = y
        if len(first_by_level) >= 2:  # 두 급 이상에서 등장
            flows.append({
                "node": node,
                "first_year_by_level": first_by_level,
                "sample": items[0][2].text[:50],
            })
    return flows


def _count(xs):
    d = defaultdict(int)
    for x in xs:
        d[x] += 1
    return d


if __name__ == "__main__":
    # python layer1.py ../data/gukeo_L1_exams.json
    exam_path = sys.argv[1] if len(sys.argv) > 1 else "../data/gukeo_L1_exams.json"
    emb = FrozenEmbedding.load("../embeddings/gukeo_emb.pkl")
    som = SOM.load("../embeddings/gukeo_som.pkl")
    exams = load_records(exam_path)
    node_exams = map_exams_to_som(exams, emb, som)

    import json
    print("=== 개념영역별 출제 연도 패턴 ===")
    print(json.dumps(concept_year_report(node_exams), ensure_ascii=False, indent=2))
    print("\n=== 급별 교차 흐름(초/중/특) ===")
    print(json.dumps(cross_level_flow(node_exams), ensure_ascii=False, indent=2))
