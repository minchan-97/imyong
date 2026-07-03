[README.md](https://github.com/user-attachments/files/29624003/README.md)
# 임용 국어 4레이어 시스템 (검증판)

한 과목(국어)으로 4개 레이어 전체를 세로로 완성한 버전.
이게 돌면 → 같은 틀을 다른 과목(수학·사회…)으로 복제해 가로 확장한다.

## 설계 핵심 (합의된 원칙)
- **임베딩은 한 번만 학습해 고정**(`embeddings/gukeo_emb.pkl`). 이전에 학습이
  '퍼진' 원인(임베딩 매번 재생성)을 차단. 모든 레이어가 같은 좌표계 공유.
- **과목별로 SOM 분리 학습**. 전과목 한 지도에 부으면 발산 → 국어만 담는다.
- **출처 태그 강제**. `source` 없는 데이터는 스키마에서 거부됨.
- **레이어4는 예측기가 아니라 연습문제 생성기.** 해설은 실제 자료 출처를
  밝혀서만 나오고, `reveal()`(버튼) 전엔 잠겨 있다(먼저 풀게 강제 = 인출연습).

## 폴더
```
core/         schema.py(출처강제) · embedding.py(고정임베딩) · som.py · korean_tokenizer.py
layer2_corpus/ layer2.py  ← 자료 매핑 + 정합성 가드레일 (여기부터 학습)
layer1_pattern/layer1.py  ← 기출 시계열 패턴 + 급별(초/중/특) 교차분석
layer3_trend/  layer3.py  ← 검색어 생성 + Brave + 가드레일 필터 (키는 로컬에서)
layer4_generate/layer4.py ← 문제 생성 + 출처 해설 + reveal 잠금
data/         gukeo_L2.json(국어자료) · gukeo_L1_exams.json(기출)
embeddings/   학습 산출물(gukeo_emb.pkl, gukeo_som.pkl)
```

## 실행 순서
```bash
# 0) (선택) 샘플로 먼저 파이프라인 확인
python make_sample_data.py

# 1) 레이어2: 임베딩 고정 학습 + 국어 SOM 학습  ← 반드시 먼저
cd layer2_corpus && python layer2.py ../data/gukeo_L2.json

# 2) 레이어1: 기출을 그 SOM에 얹어 연도·급별 패턴 분석
cd ../layer1_pattern && python layer1.py ../data/gukeo_L1_exams.json

# 3) 레이어3: 검색어 생성(→ 로컬에서 Brave 키 꽂아 검색·필터)
cd ../layer3_trend && python layer3.py

# 4) 레이어4: 개념영역별 연습문제(해설 잠금). OpenAI 키 꽂으면 실제 생성
cd ../layer4_generate && python layer4.py
```

## 데이터 업로드 규칙 (중요)
모든 자료는 아래 태그를 붙여 `Record`로 만든다. 분류 체계를 먼저 지키고 올릴 것.
- `layer`   : L1_pattern / L2_corpus / L3_trend / L4_generate
- `subject` : 국어 (지금은 국어만; 복제 시 여기만 바꿈)
- `source`  : **필수** — 파일명/책+페이지/URL 등 출처
- `year`    : 기출이면 필수(L1)
- `level`   : 초등/중등/특수/공통 (기출 교차분석용)
- `code`    : 성취기준 코드(있으면 매핑 정확도↑)

## 다른 과목으로 복제할 때
`subject`만 바꾸고(예: "수학"), 그 과목 자료로 1~2단계를 다시 돌리면
`수학_emb.pkl` / `수학_som.pkl`이 생긴다. 코드는 그대로.

## 실데이터 팁
- 임베딩 어휘가 적으면(작은 코퍼스) `min_count`를 1~2로. 자료가 커지면 2 이상.
- SOM `grid`는 자료량에 맞춰: 수백 건이면 (10~14)², 수천 건이면 더 크게.
- 레이어1 패턴은 기출이 개념영역당 2건 이상 쌓여야 보인다 → 기출을 많이 넣을수록 좋음.
```

---

## Streamlit 앱 (단일 앱·탭 방식)
```bash
pip install streamlit numpy openai
streamlit run app.py
```
사이드바에서 **과목 선택** → 4개 탭:
- 📚 **자료·학습(L2)**: 자료 입력(출처 필수) → pkl 저장 → 임베딩·SOM 학습 → 정합성 판정
- 📈 **기출 패턴(L1)**: 기출 입력(연도·급 필수) → 개념영역별 연도/급별 패턴
- 🔎 **트렌드(L3)**: 자동 검색어 → Brave 검색(키 입력) → 가드레일 필터 → pkl 저장
- 📝 **문제 풀기(L4)**: 연습문제(해설 잠금) → '🔓 해설 보기' 눌러야 근거·해설 공개

## 데이터 저장 (전부 pkl, data/ 아래)
- 원본 자료: `{과목}_L2.pkl` `{과목}_L1.pkl` `{과목}_L3.pkl`
- 학습 산출물: `{과목}_emb.pkl` `{과목}_som.pkl`
- **과목만 바꾸면** 다른 과목 데이터가 별도 pkl로 쌓인다(코드 수정 없음).

## 학습이 '퍼지지' 않게 하는 장치 (재발 방지)
1. 임베딩은 최초 1회만 학습해 고정 → 이후 모든 레이어가 같은 좌표계 공유.
2. 과목별 SOM 분리 → 전과목 혼합으로 인한 발산 차단.
3. 자료를 크게 바꿀 때만 'force_emb' 체크로 임베딩 재학습.
