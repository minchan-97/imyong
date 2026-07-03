"""
app.py — 임용 국어 4레이어 Streamlit 앱 (단일 앱·탭 방식)

실행:  streamlit run app.py

탭 구성:
  📚 자료·학습(L2)  : 성취기준/지도서 입력·업로드 → pkl 저장 → 임베딩·SOM 학습
  📈 기출 패턴(L1)  : 기출 입력 → 개념영역별 연도·급별 패턴 분석
  🔎 트렌드(L3)     : 검색어 자동 생성 → Brave 검색 → 가드레일 필터 → pkl 저장
  📝 문제 풀기(L4)  : 개념영역별 연습문제(해설 잠금) → '해설 보기' 버튼으로만 공개

데이터는 전부 과목별 pkl로 data/ 아래 저장된다.
"""
import sys, os, json
import numpy as np
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(__file__), "core"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer1_pattern"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer2_corpus"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer3_trend"))
sys.path.append(os.path.join(os.path.dirname(__file__), "layer4_generate"))

from schema import Record, save_records_pkl, load_records_pkl, LEVELS
from embedding import FrozenEmbedding, train_embedding
from som import SOM
from korean_tokenizer import tokenize
import paths

st.set_page_config(page_title="임용 4레이어", layout="wide")

# ── 사이드바: 과목 선택 (지금은 국어, 복제 시 확장) ──────────────
SUBJECT_LIST = ["국어", "영어", "수학", "사회", "과학", "미술", "음악",
                "체육", "실과", "도덕", "총론", "창의적체험활동", "통합교과"]
st.sidebar.title("⚙️ 설정")
subject = st.sidebar.selectbox("과목", SUBJECT_LIST, index=0)
st.sidebar.caption("한 과목으로 검증 후, 같은 앱에서 과목만 바꿔 확장")

# 학습 상태 표시
_has_som = paths.exists(paths.som_path(subject))
_has_emb = paths.exists(paths.emb_path(subject))
st.sidebar.markdown("---")
st.sidebar.write("**학습 상태**")
st.sidebar.write(f"임베딩: {'✅' if _has_emb else '❌ 미학습'}")
st.sidebar.write(f"SOM: {'✅' if _has_som else '❌ 미학습'}")


@st.cache_resource(show_spinner=False)
def load_engine(subject, _emb_mtime, _som_mtime):
    """학습 산출물 로드(파일 수정시각을 키로 캐시)."""
    emb = FrozenEmbedding.load(paths.emb_path(subject)) if paths.exists(paths.emb_path(subject)) else None
    som = SOM.load(paths.som_path(subject)) if paths.exists(paths.som_path(subject)) else None
    return emb, som

def _mtime(p):
    return os.path.getmtime(p) if paths.exists(p) else 0

emb, som = load_engine(subject, _mtime(paths.emb_path(subject)), _mtime(paths.som_path(subject)))

st.title(f"📖 임용 4레이어 — {subject}")

tab2, tab1, tab3, tab4 = st.tabs(
    ["📚 자료·학습 (L2)", "📈 기출 패턴 (L1)", "🔎 트렌드 (L3)", "📝 문제 풀기 (L4)"])

# ══════════════════════════════════════════════════════════════
# 탭 L2 — 자료 입력·학습
# ══════════════════════════════════════════════════════════════
with tab2:
    st.subheader("성취기준·지도서 자료 (L2)")
    l2 = load_records_pkl(paths.l2_path(subject))
    st.write(f"현재 저장된 자료: **{len(l2)}건**")

    # ── 파일 업로드(pdf/docx/txt) ────────────────────────────
    with st.expander("📄 파일로 자료 넣기 (pdf / docx / txt)", expanded=False):
        from file_ingest import ingest
        up = st.file_uploader("성취기준·지도서 파일", type=["pdf", "docx", "txt"],
                              key="l2_file")
        fsrc = st.text_input("이 파일의 출처(필수)", key="l2_fsrc",
                             placeholder="2022개정 국어과 교육과정")
        if up is not None:
            try:
                items = ingest(up.name, up.read())
                st.write(f"**추출된 항목 {len(items)}개** — 넣을 것만 체크하고 수정하세요")
                # 편집 테이블: 체크/문장/코드
                import pandas as pd
                df = pd.DataFrame([{"넣기": True, "문장": t, "코드": c or ""}
                                   for t, c in items])
                edited = st.data_editor(df, use_container_width=True,
                                        num_rows="dynamic", key="l2_editor")
                if st.button("체크한 항목 저장", type="primary", key="l2_file_save"):
                    if not fsrc.strip():
                        st.error("출처는 필수입니다.")
                    else:
                        added = 0
                        for _, row in edited.iterrows():
                            if not row["넣기"] or not str(row["문장"]).strip():
                                continue
                            try:
                                l2.append(Record(
                                    text=str(row["문장"]).strip(), layer="L2_corpus",
                                    subject=subject, source=fsrc.strip(),
                                    code=str(row["코드"]).strip() or None))
                                added += 1
                            except Exception as e:
                                st.warning(f"거부: {str(row['문장'])[:20]} — {e}")
                        save_records_pkl(l2, paths.l2_path(subject))
                        st.success(f"{added}건 저장 → {subject}_L2.pkl")
                        st.rerun()
            except Exception as e:
                st.error(f"추출 실패: {e}")

    with st.expander("➕ 직접 입력", expanded=(len(l2) == 0)):
        c1, c2 = st.columns([3, 1])
        txt = c1.text_area("자료 문장(한 줄에 하나)", height=120,
                           placeholder="이야기를 읽고 인물의 마음을 짐작하며 감상한다")
        src = c2.text_input("출처(필수)", placeholder="2022개정 국어과 성취기준")
        code = c2.text_input("성취기준 코드(선택)", placeholder="[4국05-04]")
        if st.button("자료 저장", type="primary"):
            if not src.strip():
                st.error("출처는 필수입니다.")
            else:
                added = 0
                for line in txt.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        l2.append(Record(text=line, layer="L2_corpus",
                                         subject=subject, source=src.strip(),
                                         code=code.strip() or None))
                        added += 1
                    except Exception as e:
                        st.warning(f"거부됨: {line[:20]} — {e}")
                save_records_pkl(l2, paths.l2_path(subject))
                st.success(f"{added}건 저장 → {subject}_L2.pkl")
                st.rerun()

    if l2:
        st.dataframe([{"내용": r.text[:40], "출처": r.source, "코드": r.code}
                      for r in l2[-20:]], use_container_width=True)

    st.markdown("---")
    st.subheader("🧠 임베딩 + SOM 학습")
    st.caption("임베딩은 한 번 학습해 고정. 자료를 크게 바꿨을 때만 다시 학습.")
    cc1, cc2, cc3 = st.columns(3)
    dim = cc1.number_input("임베딩 차원", 16, 128, 32, step=16)
    min_count = cc2.number_input("최소 등장(min_count)", 1, 5, 1)
    grid = cc3.number_input("SOM 격자(한 변)", 6, 20, 10)
    force_emb = st.checkbox("임베딩도 다시 학습(자료 크게 바뀜)", value=not _has_emb)

    if st.button("학습 시작", type="primary"):
        if len(l2) < 3:
            st.error("자료가 너무 적습니다(최소 3건).")
        else:
            texts = [r.text for r in l2]
            with st.spinner("임베딩 학습 중..." if force_emb else "임베딩 로드..."):
                if force_emb or not _has_emb:
                    e = train_embedding(texts, dim=int(dim),
                                        min_count=int(min_count), epochs=30)
                    e.save(paths.emb_path(subject))
                else:
                    e = FrozenEmbedding.load(paths.emb_path(subject))
            X, kept = [], []
            for r in l2:
                v = e.embed_tokens(tokenize(r.text))
                if v is not None:
                    X.append(v); kept.append(r)
            if not X:
                st.error("벡터화 실패(모르는 단어뿐). min_count를 1로.")
            else:
                with st.spinner("SOM 학습 중..."):
                    s = SOM(grid=(int(grid), int(grid)), dim=e.dim)
                    s.train(np.array(X), iters=4000)
                    s.assign(np.array(X), kept)
                    s.save(paths.som_path(subject))
                st.success(f"학습 완료! 어휘 {len(e.word2idx)}개, 벡터화 {len(kept)}/{len(l2)}건")
                load_engine.clear()
                st.rerun()

    # 정합성 판정 데모
    if som is not None and emb is not None:
        st.markdown("---")
        st.subheader("✅ 자료 정합성 판정 (가드레일)")
        from layer2 import judge
        q = st.text_input("판정할 문장", placeholder="학생이 인물의 마음을 짐작하여 표현한다")
        if q:
            by_id = {r.rec_id: r for r in l2}
            res = judge(q, emb, som, by_id)
            v = res.get("verdict")
            if v == "정합":
                st.success(f"정합 (유사도 {res['score']})")
            elif v == "판정불가":
                st.warning(res.get("reason"))
            else:
                st.error(v + f" (유사도 {res.get('score')})")
            if res.get("concept_sources"):
                st.write("**근거 개념영역 자료(출처):**")
                st.json(res["concept_sources"])

# ══════════════════════════════════════════════════════════════
# 탭 L1 — 기출 패턴
# ══════════════════════════════════════════════════════════════
with tab1:
    st.subheader("기출 시계열 패턴 (L1)")
    st.caption("⚠️ 예측기 아님 — 어느 개념이 어느 해에 몰렸나 보는 '경향 분석'")
    l1 = load_records_pkl(paths.l1_path(subject))
    st.write(f"저장된 기출: **{len(l1)}건**")

    # ── 기출 파일 업로드 (초등: 통짜 시험지 → 문항 분할) ──────
    with st.expander("📄 기출 파일로 넣기 (pdf / docx / txt)", expanded=False):
        from file_ingest import extract_text, split_questions_rule, \
            split_questions_llm, is_scanned_pdf
        st.caption("초등은 한 시험에 전과목이 섞임 → 연도 단위로 통짜 업로드하면 "
                   "문항으로 잘라준다. 중등/특수는 과목별 시험이라 그대로.")
        upq = st.file_uploader("기출 시험지 파일", type=["pdf", "docx", "txt"], key="l1_file")
        fc1, fc2, fc3 = st.columns(3)
        fyear = fc1.number_input("출제연도", 2000, 2030, 2023, key="l1_fy")
        flevel = fc2.selectbox("급", ["초등", "중등", "특수", "공통"], key="l1_fl")
        fqsrc = fc3.text_input("출처(필수)", key="l1_fs", placeholder="2023 초등임용")

        use_llm_split = st.checkbox("LLM으로 문항 분할(더 정확, OpenAI 키 필요)",
                                    key="l1_llmsplit")
        split_key = ""
        if use_llm_split:
            split_key = st.text_input("OpenAI Key(분할용)", type="password", key="l1_splitkey")

        if upq is not None:
            try:
                raw_text = extract_text(upq.name, upq.read())
                if is_scanned_pdf(raw_text):
                    st.error("이 PDF는 텍스트가 추출되지 않습니다(스캔본으로 보임). "
                             "OCR이 필요해요 — 텍스트 PDF로 다시 저장하거나 OCR 후 넣어주세요.")
                else:
                    if use_llm_split and split_key:
                        items = split_questions_llm(raw_text, split_key)
                    else:
                        items = split_questions_rule(raw_text)
                    st.write(f"**분할된 문항 후보 {len(items)}개** — "
                             "제목·안내문은 체크 해제, 묶음문항은 행을 나눠서 편집")
                    import pandas as pd
                    df = pd.DataFrame([{"넣기": True, "문항": t} for t in items])
                    edited = st.data_editor(df, use_container_width=True,
                                            num_rows="dynamic", key="l1_editor")
                    st.caption("💡 초등 통짜 기출은 지금 subject='{}'로 저장됩니다. "
                               "과목 자동분류는 다른 과목 SOM이 갖춰지면 붙일 수 있어요."
                               .format(subject))
                    if st.button("체크한 문항 저장", type="primary", key="l1_file_save"):
                        if not fqsrc.strip():
                            st.error("출처는 필수입니다.")
                        else:
                            added = 0
                            for _, row in edited.iterrows():
                                if not row["넣기"] or not str(row["문항"]).strip():
                                    continue
                                try:
                                    l1.append(Record(
                                        text=str(row["문항"]).strip(), layer="L1_pattern",
                                        subject=subject, source=fqsrc.strip(),
                                        year=int(fyear), level=flevel))
                                    added += 1
                                except Exception as e:
                                    st.warning(f"거부: {str(row['문항'])[:20]} — {e}")
                            save_records_pkl(l1, paths.l1_path(subject))
                            st.success(f"{added}건 저장 → {subject}_L1.pkl")
                            st.rerun()
            except Exception as e:
                st.error(f"추출/분할 실패: {e}")

    with st.expander("➕ 직접 입력", expanded=(len(l1) == 0)):
        qtxt = st.text_area("기출 문항", height=80,
                            placeholder="인물의 마음을 짐작하는 지도 방법을 서술하시오")
        d1, d2, d3 = st.columns(3)
        year = d1.number_input("출제연도(필수)", 2000, 2030, 2023)
        level = d2.selectbox("급(필수)", ["초등", "중등", "특수", "공통"])
        qsrc = d3.text_input("출처(필수)", placeholder="2023 초등임용 국어")
        if st.button("기출 저장", type="primary", key="save_l1"):
            if not qtxt.strip() or not qsrc.strip():
                st.error("문항·출처는 필수입니다.")
            else:
                try:
                    l1.append(Record(text=qtxt.strip(), layer="L1_pattern",
                                     subject=subject, source=qsrc.strip(),
                                     year=int(year), level=level))
                    save_records_pkl(l1, paths.l1_path(subject))
                    st.success(f"저장 → {subject}_L1.pkl")
                    st.rerun()
                except Exception as e:
                    st.error(str(e))

    if som is None or emb is None:
        st.info("먼저 L2 탭에서 학습을 완료하세요(기출을 얹을 지도가 필요).")
    elif l1:
        from layer1 import map_exams_to_som, concept_year_report, cross_level_flow
        node_exams = map_exams_to_som(l1, emb, som)
        min_hits = st.slider("패턴 최소 반복", 1, 5, 2)
        st.write("**개념영역별 출제 연도**")
        rep = concept_year_report(node_exams, min_hits=min_hits)
        if rep:
            st.dataframe([{"개념샘플": r["sample"], "출제수": r["hit_count"],
                           "연도": r["years"], "급별": r["levels"]} for r in rep],
                         use_container_width=True)
        else:
            st.caption("아직 반복 패턴 없음 — 기출을 더 넣으면 같은 개념에 연도가 쌓임")
        flows = cross_level_flow(node_exams)
        if flows:
            st.write("**급별 교차 흐름(초/중/특 등장 시점)**")
            st.json(flows)

# ══════════════════════════════════════════════════════════════
# 탭 L3 — 트렌드 검색
# ══════════════════════════════════════════════════════════════
with tab3:
    st.subheader("최신 트렌드/논문 검색 (L3)")
    if som is None or emb is None:
        st.info("먼저 L2 학습을 완료하세요.")
    else:
        from layer3 import make_queries, brave_search, filter_by_guardrail
        st.write("**‘자료가 성긴 영역’ 기반 자동 검색어**")
        qs = make_queries(emb, som, subject=subject)
        st.dataframe([{"검색어": q["query"], "QE": round(q["qe"], 3)} for q in qs],
                     use_container_width=True)
        key = st.text_input("Brave API Key", type="password")
        pass_thr = st.slider("도메인 필터 강도", 0.2, 0.7, 0.40, 0.05)
        if st.button("검색 + 필터 실행") and key:
            l3 = load_records_pkl(paths.l3_path(subject))
            got = 0
            for q in qs[:5]:
                try:
                    results = brave_search(q["query"], key)
                    kept = filter_by_guardrail(results, emb, som, subject, pass_thr)
                    l3.extend(kept); got += len(kept)
                except Exception as e:
                    st.warning(f"검색 실패: {q['query'][:20]} — {e}")
            save_records_pkl(l3, paths.l3_path(subject))
            st.success(f"필터 통과 {got}건 저장 → {subject}_L3.pkl")

# ══════════════════════════════════════════════════════════════
# 탭 L4 — 문제 풀기 (해설 잠금 → 버튼으로만 공개)
# ══════════════════════════════════════════════════════════════
with tab4:
    st.subheader("연습문제 (L4)")
    st.caption("문제 풀기 → 내 답 채점(제안) → 내가 확정 → 약점지도 학습. 문제·해설 검토로 함께 발전.")
    if som is None or emb is None:
        st.info("먼저 L2 학습을 완료하세요.")
    else:
        from layer4 import pick_concept_nodes, gather_grounding, \
            build_generation_prompt, generate_with_llm, \
            grade_answer_llm, grade_answer_offline
        from study_state import StudyState
        l2 = load_records_pkl(paths.l2_path(subject))
        l1 = load_records_pkl(paths.l1_path(subject))
        by_id = {r.rec_id: r for r in (l2 + l1)}
        study = StudyState.load(paths.study_path(subject), subject)

        # ── 약점 지도 표시 ──────────────────────────────────
        with st.expander("📊 내 약점 지도", expanded=False):
            wc = study.weak_by_code()
            wn = study.weak_by_node()
            if wc:
                st.write("**성취기준 코드별 (정답률 낮은 순)**")
                st.dataframe(wc, use_container_width=True)
            if wn:
                st.write("**개념영역별**")
                st.dataframe(wn, use_container_width=True)
            if not wc and not wn:
                st.caption("아직 채점 기록 없음 — 문제를 풀고 답을 확정하면 쌓입니다.")

        okey = st.text_input("OpenAI API Key (문제·채점·해설용)", type="password")
        n = st.slider("문제 수", 1, 8, 3)
        target_weak = st.checkbox("내 약점 영역 위주로 출제", value=bool(study.weak_by_node()))

        if st.button("문제 생성", type="primary"):
            from layer1 import map_exams_to_som
            l1_nodes = map_exams_to_som(l1, emb, som) if l1 else None
            if target_weak and study.weak_by_node():
                nodes = study.weak_nodes_for_targeting(top=n)
            else:
                nodes = pick_concept_nodes(som, l1_nodes, topn=n)
            probs = []
            for node in nodes:
                # 신뢰도 낮은 자료는 근거에서 후순위 (trust 반영)
                grounding = gather_grounding(node, som, by_id)
                grounding.sort(key=lambda g: 0)  # 순서 유지(자리표시)
                if okey:
                    prompt = build_generation_prompt(grounding)
                    try:
                        stem, answer = generate_with_llm(prompt, okey)
                    except Exception as e:
                        stem, answer = f"(생성 실패: {e})", ""
                else:
                    stem = f"[개념영역 {node} — OpenAI 키 넣으면 실제 문제 생성]"
                    answer = None
                # 이 개념영역의 대표 코드(약점지도 기록용)
                codes = som.node_codes.get(node, [])
                probs.append({"node": node, "stem": stem, "answer": answer,
                              "grounding": grounding, "codes": list(set(codes))})
            st.session_state["problems"] = probs
            st.session_state["revealed"] = set()
            st.session_state["graded"] = {}

        # ── 문제별: 풀기 → 채점 → 확정 → 검토 ──────────────
        for i, p in enumerate(st.session_state.get("problems", [])):
            st.markdown(f"### 문제 {i+1}  ·  개념영역 {p['node']}")
            st.write(p["stem"])

            my_ans = st.text_area("✍️ 내 답", key=f"ans_{i}", height=100)

            cc1, cc2, cc3 = st.columns(3)
            # 1) 자동 채점 제안
            if cc1.button("🤖 채점 제안", key=f"grade_{i}"):
                if not my_ans.strip():
                    st.warning("답을 먼저 쓰세요.")
                else:
                    if okey:
                        g = grade_answer_llm(p["stem"], my_ans, p["grounding"], okey)
                    else:
                        g = grade_answer_offline(my_ans, emb, som, p["node"])
                    st.session_state.setdefault("graded", {})[i] = g
                    st.rerun()

            g = st.session_state.get("graded", {}).get(i)
            if g:
                label = {"correct": "✅ 맞을 것 같음", "partial": "🟡 부분/애매",
                         "wrong": "❌ 틀린 것 같음"}.get(g.get("suggest"), "🟡")
                st.info(f"**자동 제안: {label}** (참고용)\n\n{g.get('feedback','')}")

            # 2) 내가 최종 확정 (이것만 약점지도 반영)
            st.write("**내 확정** (이게 약점지도에 반영됨)")
            fc1, fc2 = st.columns(2)
            if fc1.button("맞음으로 확정", key=f"ok_{i}"):
                study.record_answer(p["node"], p["codes"], my_ans,
                                    (g or {}).get("suggest"), "correct")
                study.save(paths.study_path(subject))
                st.success("맞음으로 기록됨 → 약점지도 갱신")
            if fc2.button("틀림으로 확정", key=f"no_{i}"):
                study.record_answer(p["node"], p["codes"], my_ans,
                                    (g or {}).get("suggest"), "wrong")
                study.save(paths.study_path(subject))
                st.error("틀림으로 기록됨 → 약점지도 갱신")

            # 3) 해설 보기 (잠금 → 버튼)
            revealed = st.session_state.get("revealed", set())
            if i in revealed:
                st.success("**해설**")
                model_ans = (g or {}).get("model_answer") or p["answer"] or "(LLM 미연결)"
                st.write(model_ans)
                st.write("**📎 근거 출처**")
                st.json(p["grounding"])
                # 4) 문제·해설 검토 (자료 신뢰도 조정 + 복구)
                st.write("**🔧 이 문제·해설 검토** (자료 개선에 반영)")
                rc1, rc2, rc3 = st.columns(3)
                if rc1.button("👍 좋은 문제", key=f"good_{i}"):
                    for g_ in p["grounding"]:
                        pass  # grounding엔 rec_id 없음 → 노드 자료로 반영
                    for rid in som.node_rec_ids.get(p["node"], [])[:3]:
                        study.review_feedback(rid, good=True, reason=f"문제{i+1} 좋음")
                    study.save(paths.study_path(subject))
                    st.success("좋은 자료로 반영")
                if rc2.button("👎 이상한 문제/해설", key=f"bad_{i}"):
                    for rid in som.node_rec_ids.get(p["node"], [])[:3]:
                        study.review_feedback(rid, good=False, reason=f"문제{i+1} 이상")
                    study.save(paths.study_path(subject))
                    st.warning("해당 개념영역 자료 신뢰도 하향(복구 가능)")
                if rc3.button("↩️ 방금 검토 복구", key=f"undo_{i}"):
                    u = study.undo_last_review()
                    study.save(paths.study_path(subject))
                    st.info(f"복구됨: {u['rec_id'][:8] if u else '없음'}")
            else:
                if st.button("🔓 해설 보기", key=f"reveal_{i}"):
                    st.session_state["revealed"].add(i)
                    st.rerun()
            st.markdown("---")
