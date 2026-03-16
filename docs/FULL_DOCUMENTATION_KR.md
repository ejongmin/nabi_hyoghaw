# 나비효과(nabi_hyoghaw) — EV-배터리 공급망 리스크 분석 MVP 전체 문서

> **목표**: 뉴스(GDELT) → 리스크 이벤트 → 공급망 그래프(KG) → 노출도(Exposure) → CAR → 백테스트 → 리포트

---

## 목차

1. [프로젝트 개요](#1-프로젝트-개요)
2. [디렉토리 구조](#2-디렉토리-구조)
3. [설치 및 실행](#3-설치-및-실행)
4. [파이프라인 흐름](#4-파이프라인-흐름)
5. [모듈 상세 설명](#5-모듈-상세-설명)
   - 5.1 [공통 유틸리티 (src/common/)](#51-공통-유틸리티-srccommon)
   - 5.2 [데이터 수집 (src/ingest/)](#52-데이터-수집-srcingest)
   - 5.3 [NLP 처리 (src/nlp/)](#53-nlp-처리-srcnlp)
   - 5.4 [지식 그래프 (src/kg/)](#54-지식-그래프-srckg)
   - 5.5 [그래프 분석 (src/graph/)](#55-그래프-분석-srcgraph)
   - 5.6 [GNN 모델 (src/gnn/)](#56-gnn-모델-srcgnn)
   - 5.7 [재무 분석 (src/finance/)](#57-재무-분석-srcfinance)
   - 5.8 [데이터 품질 (src/quality/)](#58-데이터-품질-srcquality)
   - 5.9 [리포트 생성 (src/reporting/)](#59-리포트-생성-srcreporting)
   - 5.10 [보조 도구 (src/tools/)](#510-보조-도구-srctools)
6. [CLI 명령어 레퍼런스](#6-cli-명령어-레퍼런스)
7. [설정 파일 상세](#7-설정-파일-상세)
8. [핵심 데이터 모델](#8-핵심-데이터-모델)
9. [테스트](#9-테스트)
10. [CI/CD](#10-cicd)
11. [외부 의존성](#11-외부-의존성)

---

## 1. 프로젝트 개요

**나비효과(nabi_hyoghaw)**는 EV(전기차)-배터리 공급망에서 발생하는 지정학·물류·기후 리스크가 개별 기업 주가에 미치는 영향을 정량적으로 분석하는 파이프라인입니다.

### 핵심 아이디어

1. **GDELT 뉴스**에서 리스크 이벤트(제재, 항만 파업, 지진 등)를 탐지
2. 뉴스 기사를 **기업 엔티티**에 매칭(Entity Linking)
3. 사전 구축된 **공급망 지식 그래프(KG)**를 통해 리스크가 어떤 기업으로 전파되는지 계산(Exposure)
4. **이벤트 스터디(CAR)**로 주가 영향을 측정
5. 리스크 회피 전략의 **백테스트**로 투자 관점 유효성 검증

### 기술 스택

| 영역 | 주요 기술 |
|------|-----------|
| 데이터 수집 | GDELT DOC API, yfinance |
| NLP | 키워드 분류, FinBERT 감성 분석 |
| 그래프 | NetworkX (최단경로, RWR), PyG (GAT) |
| 엔티티 매칭 | rapidfuzz (퍼지 매칭) |
| 재무 분석 | statsmodels (OLS), 자체 백테스트 엔진 |
| CLI | Typer + Rich |
| 데이터 | Pandas, PyArrow (Parquet) |

---

## 2. 디렉토리 구조

```
nabi_hyoghaw/
├── src/                              # 핵심 소스코드
│   ├── cli.py                        #   CLI 진입점 (Typer 앱)
│   ├── common/                       #   공통 유틸리티
│   │   ├── config.py                 #     YAML 설정 로더 (Config 클래스)
│   │   ├── dates.py                  #     날짜 변환 헬퍼
│   │   ├── io.py                     #     멀티포맷 DataFrame I/O
│   │   ├── log.py                    #     로깅 설정 (Rich 핸들러)
│   │   └── md.py                     #     DataFrame → 마크다운 변환
│   ├── ingest/                       #   데이터 수집 모듈
│   │   ├── gdelt.py                  #     GDELT DOC API 수집
│   │   ├── gdelt_fast_processor.py   #     대용량 GDELT CSV 청크 처리
│   │   ├── prices.py                 #     yfinance 주가 수집
│   │   ├── tickers.py                #     유니버스 → 티커 매핑
│   │   └── mock.py                   #     오프라인 테스트용 목 데이터
│   ├── nlp/                          #   NLP 처리 모듈
│   │   ├── risk_scoring.py           #     리스크 이벤트 추출 & 심각도 계산
│   │   ├── entity_linking.py         #     기사 ↔ 기업 엔티티 매칭
│   │   └── finbert.py                #     FinBERT 감성 분석 (선택)
│   ├── kg/                           #   지식 그래프 구축
│   │   ├── build_static.py           #     유니버스 + seed → 정적 KG
│   │   └── llm_extractor.py          #     LLM 기반 관계 추출 (실험적)
│   ├── graph/                        #   그래프 분석
│   │   └── baseline_exposure.py      #     최단경로 & RWR 기반 노출도
│   ├── gnn/                          #   그래프 신경망
│   │   └── gat.py                    #     GAT (Graph Attention Network)
│   ├── finance/                      #   재무 분석
│   │   ├── event_study.py            #     이벤트 스터디 (CAR 계산)
│   │   ├── backtest.py               #     리스크 회피 백테스트
│   │   └── metrics.py                #     MDD, CAGR, Sharpe 등 지표
│   ├── quality/                      #   데이터 품질 관리
│   │   ├── dq_gate.py                #     통합 DQ 리포트 생성
│   │   ├── entity_qc.py              #     엔티티 링킹 성공률 점검
│   │   ├── kg_qc.py                  #     KG 엣지 유효성 점검
│   │   ├── news_qc.py                #     뉴스 중복 제거 & 점검
│   │   └── prices_qc.py              #     주가 결측/이상치 점검
│   ├── reporting/                    #   리포트 생성
│   │   └── report_all.py             #     마크다운 리포트 인덱스
│   └── tools/                        #   보조 도구
│       ├── env_doctor.py             #     환경/의존성 점검
│       ├── seed_from_xlsx.py         #     Excel → seed_edges 변환
│       ├── seed_from_docx.py         #     DOCX → seed_edges 변환
│       ├── seed_from_folder.py       #     폴더 일괄 시드 임포트
│       ├── universe_utils.py         #     유니버스 정규화 & 티커 관리
│       ├── validators.py             #     유니버스/시드 검증
│       └── price_converter.py        #     원시 주가 CSV → Parquet 변환
│
├── configs/                          # 설정 파일
│   ├── base.yaml                     #   기본 파이프라인 설정
│   ├── base_colab_drive.yaml         #   Colab/Google Drive용 설정
│   ├── risk_keywords.yaml            #   리스크 키워드 사전
│   ├── schema.yaml                   #   KG 노드/관계 스키마
│   ├── seed_mapping.yaml             #   Excel/DOCX 임포트 컬럼 매핑
│   └── universe_priority.yaml        #   티커 거래소 우선순위
│
├── data/                             # 데이터 디렉토리
│   ├── seed/                         #   시드 데이터 (seed_edges.csv, 템플릿)
│   ├── universe/                     #   기업 유니버스 (univers_final.csv)
│   ├── raw/gdelt/                    #   원본 GDELT 뉴스 데이터
│   └── processed/                    #   처리 결과 (parquet/csv)
│
├── tests/                            # 단위 테스트
├── notebooks/                        # Colab 노트북
├── docs/                             # 문서
├── reports/                          # 생성된 리포트 (MD)
├── .github/workflows/                # CI/CD (smoke-test)
├── Makefile                          # Make 명령어
├── requirements.txt                  # 코어 의존성
├── requirements_nlp.txt              # FinBERT 의존성 (선택)
└── requirements_gnn.txt              # GAT 의존성 (선택)
```

---

## 3. 설치 및 실행

### 3.1 로컬 설치

```bash
# 기본 설치
python -m pip install -r requirements.txt

# (선택) FinBERT 감성 점수
python -m pip install -r requirements_nlp.txt

# (선택) GAT 그래프 신경망
python -m pip install -r requirements_gnn.txt
```

### 3.2 빠른 실행

```bash
# 디렉토리 초기화 + 전체 파이프라인
python -m src.cli init --config configs/base.yaml
python -m src.cli pipeline --config configs/base.yaml

# 오프라인 스모크 테스트 (목 데이터 사용)
python -m src.cli smoke-test --config configs/base.yaml
```

### 3.3 Makefile 활용

```bash
make setup      # 의존성 설치
make pipeline   # 전체 파이프라인 실행
make smoke      # 스모크 테스트
```

---

## 4. 파이프라인 흐름

```
┌─────────────────────────────────────────────────────────────────┐
│                        전체 파이프라인                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  [0] init / doctor          디렉토리 생성 & 환경 점검            │
│       ↓                                                         │
│  [1] normalize-universe     유니버스 정규화 (티커 선택/정리)      │
│       ↓                                                         │
│  [2] make-tickers           yfinance 티커 매핑 생성              │
│       ↓                                                         │
│  [3] ingest-gdelt           GDELT 뉴스 기사 수집                 │
│  [4] ingest-prices          주가 데이터 수집 (yfinance)          │
│       ↓                                                         │
│  [5] build-risk-events      NLP로 리스크 이벤트 추출             │
│       ↓                                                         │
│  [6] build-kg               공급망 지식 그래프 구축              │
│       ↓                                                         │
│  [7] compute-exposure       리스크 노출도 계산                   │
│       │                     (최단경로 + RWR + GAT)              │
│       ↓                                                         │
│  [8] event-study            CAR 이벤트 스터디                    │
│  [9] backtest               리스크 회피 백테스트                  │
│       ↓                                                         │
│  [10] data-quality          데이터 품질 리포트                   │
│  [11] report-all            최종 리포트 인덱스 생성              │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 데이터 흐름도

```
GDELT API ──→ articles.csv ──→ risk_events.parquet ──┐
                                                      │
universe.csv ──→ universe_normalized.csv              │
    │                                                  │
    ├──→ ticker_map.parquet ──→ prices.parquet         │
    │                              │                   │
seed_edges.csv ──→ kg_nodes.parquet                   │
                   kg_edges.parquet ──→ exposure.parquet
                                           │
                              prices ──→ car_panel.parquet
                                         backtest_equity.csv
                                              │
                                         reports/*.md
```

---

## 5. 모듈 상세 설명

---

### 5.1 공통 유틸리티 (src/common/)

#### config.py — 설정 관리

YAML 기반 설정을 로드하고, 경로를 해석하는 핵심 클래스입니다.

```python
class Config:
    raw: Dict[str, Any]       # 원본 YAML 딕셔너리
    config_path: Path         # 설정 파일 경로
    root_dir: Path            # 프로젝트 루트 디렉토리
```

| 메서드 | 설명 |
|--------|------|
| `Config.load(path)` | YAML 파일을 읽어 Config 객체 생성. `root_dir`은 설정 내 `project.root_dir` 또는 config 파일의 부모 디렉토리 |
| `get(*keys, default=None)` | 중첩 딕셔너리 접근. 예: `cfg.get("risk", "keywords_yaml")` → `cfg.raw["risk"]["keywords_yaml"]` |
| `path(*keys)` | `cfg.raw["paths"][key]`를 `root_dir` 기준 절대경로로 변환 |

```python
ensure_dirs(*paths: Path) -> None
```
- 주어진 경로들의 디렉토리를 재귀적으로 생성 (`mkdir(parents=True, exist_ok=True)`)

---

#### dates.py — 날짜 처리

```python
to_timestamp(x) -> Optional[pd.Timestamp]
```
- 다양한 포맷(문자열, datetime, int 등)을 `pd.Timestamp`로 변환
- 변환 실패 시 `None` 반환

```python
next_trading_day(trading_days: np.ndarray, event_day: pd.Timestamp) -> Optional[pd.Timestamp]
```
- 정렬된 거래일 배열에서 `event_day` 이후 첫 번째 거래일을 이진 탐색으로 반환
- 이벤트 스터디에서 이벤트 발생일 → 실제 거래 시작일을 찾을 때 사용

---

#### io.py — 데이터 I/O

```python
read_df(path: Path) -> pd.DataFrame
```
- 지원 포맷: `.parquet`, `.csv`, `.tsv`, `.xlsx`, `.xls`
- 파일 미존재 시 `FileNotFoundError`, 미지원 포맷 시 `ValueError`

```python
write_df(df: pd.DataFrame, path: Path, index: bool = False) -> None
```
- 지원 포맷: `.parquet`, `.csv`, `.tsv`
- 부모 디렉토리 자동 생성, CSV/TSV는 `UTF-8-sig` 인코딩

---

#### log.py — 로깅

```python
setup_logging(level: str = "INFO") -> None   # Rich 핸들러 기반 로깅 초기화
get_logger(name: str = "nabi") -> Logger      # 이름 지정 로거 반환
```

---

#### md.py — 마크다운 헬퍼

```python
df_to_markdown(df: pd.DataFrame, max_rows: int = 20) -> str
```
- DataFrame을 마크다운 테이블로 변환 (최대 `max_rows`행)
- `tabulate` 미설치 시 CSV 폴백

---

### 5.2 데이터 수집 (src/ingest/)

#### gdelt.py — GDELT 뉴스 수집

```python
ingest_gdelt(cfg: Dict, out_csv: Path, force: bool = False) -> pd.DataFrame
```

GDELT DOC API를 통해 뉴스 기사를 수집합니다.

| 파라미터 | 설명 |
|----------|------|
| `cfg` | `gdelt.*`, `project.*` 설정이 포함된 딕셔너리 |
| `out_csv` | 출력 CSV 경로 (캐시: 존재하면 스킵, `force=True`로 강제 재수집) |

**동작 과정:**
1. `gdeltdoc.Filters`로 검색 조건 구성 (키워드, 날짜 범위, 언어)
2. `gdeltdoc.GdeltDoc().article_search()` 호출
3. 실패 시 지수 백오프 재시도 (최대 `retries`회)
4. `max_records` 제한 적용
5. CSV 저장 (UTF-8-sig)

**설정 예시 (base.yaml):**
```yaml
gdelt:
  language: english
  keyword_query: '"export ban" OR "sanction" OR "tariff" ...'
  max_records: 5000
  retries: 3
  retry_backoff_sec: 2.0
```

---

#### gdelt_fast_processor.py — 대용량 GDELT 처리

```python
fast_process_gdelt_to_parquet(
    csv_path: Path,
    universe_path: Path,
    output_parquet: Path,
    keywords_yaml: Path,
    chunk_size: int = 50000
) -> None
```

BigQuery 등에서 받은 대용량 GDELT CSV를 청크 단위로 처리합니다.

**처리 단계:**
1. 유니버스 로드 & `EntityLinker` 초기화 (min_score=90)
2. 리스크 키워드 분류기 로드
3. CSV를 `chunk_size` 행 단위로 읽으며:
   - `event_date` → `event_time` 표준화
   - `Actor1Name`/`Actor2Name` 기반 엔티티 링킹
   - GoldsteinScale(-10~+10), AvgTone(-100~+100)으로 심각도(severity) 계산
   - 리스크 타입 분류 (키워드 매칭)
   - `event_id` 생성 (URL + datetime의 MD5 해시)
4. Parquet에 점진적 저장

**심각도 계산 공식:**
```
norm_gs = (-GoldsteinScale + 10) / 20    # -10→1.0, +10→0.0
norm_at = (-AvgTone + 100) / 200
severity = clip((norm_gs × 2.5) + (norm_at × 1.5) + 1.0, 1.0, 5.0)
```

---

#### prices.py — 주가 수집

```python
ingest_prices_from_universe(
    universe: pd.DataFrame,
    rules: Dict[str, str],
    out_parquet: Path,
    start: str, end: str,
    retries: int = 3,
    backoff: float = 2.0,
    force: bool = False,
    batch_size: int = 30
) -> pd.DataFrame
```

유니버스의 모든 기업에 대해 yfinance로 주가를 수집합니다.

**동작 과정:**
1. 유니버스 `company_id` → yfinance 티커 매핑 생성 (정규화 규칙 적용)
2. `batch_size`개씩 묶어 다운로드 (지수 백오프 재시도)
3. MultiIndex 결과 처리 → tidy 포맷 변환
4. `provider_ticker` → `company_id` 역매핑
5. Parquet 저장 + `missing_tickers.txt` 생성 (매핑 실패 티커 목록)

**출력 컬럼:** `date`, `company_id`, `provider_ticker`, `close`, `adj_close`, `volume`

---

#### tickers.py — 티커 매핑

```python
normalize_for_yfinance(ticker: str, rules: Dict[str, str]) -> str
```
- 티커 접미사 변환 규칙 적용. 예: `.SH` → `.SS` (상하이 증권거래소)

```python
build_ticker_map(universe: pd.DataFrame, rules: Dict[str, str]) -> pd.DataFrame
```
- 유니버스의 `company_id` → yfinance `provider_ticker` 매핑 테이블 생성

---

#### mock.py — 목 데이터 생성

```python
make_mock_articles(out_csv: Path) -> pd.DataFrame
```
- 하드코딩된 3개 뉴스 기사 생성 (오프라인 테스트용)

```python
make_mock_prices(company_ids: list, out_parquet: Path, start: str, end: str) -> pd.DataFrame
```
- 랜덤 워크 기반 합성 주가 생성 (seed=42로 재현 가능)
- 영업일 기준 날짜 범위, 시작가 100 + 정규분포 누적합

---

### 5.3 NLP 처리 (src/nlp/)

#### risk_scoring.py — 리스크 이벤트 추출

이 모듈은 뉴스 기사를 리스크 이벤트로 변환하는 파이프라인의 핵심입니다.

**RiskKeywordModel 클래스:**
```python
class RiskKeywordModel:
    keyword_map: Dict[str, List[str]]   # 리스크 유형 → 키워드 리스트

    @staticmethod
    def load(path: Path) -> "RiskKeywordModel"   # YAML에서 로드
    def classify(self, text: str) -> List[str]    # 텍스트 → 리스크 유형 분류
```

**build_risk_events 함수:**
```python
build_risk_events(
    articles: pd.DataFrame,      # 뉴스 기사 (seendate, title, url)
    universe: pd.DataFrame,      # 기업 유니버스
    keywords_yaml: Path,         # risk_keywords.yaml 경로
    cfg_risk: Dict,              # 리스크 설정
    cfg_link: Dict               # 엔티티 링킹 설정
) -> pd.DataFrame
```

**처리 단계:**

| 단계 | 설명 |
|------|------|
| 1. 컬럼 표준화 | `seendate`/`title`/`url` 컬럼명 통일 |
| 2. 리스크 분류 | 기사 제목을 키워드로 분류 → `risk_types` (쉼표 구분) |
| 3. FinBERT (선택) | `use_finbert=True`이면 감성 점수 추가 |
| 4. 심각도 계산 | `severity = base + keyword_weight × 키워드수 + finbert_weight × 감성` |
| 5. 엔티티 링킹 | `EntityLinker`로 기사 → 기업 매칭 |
| 6. 필터링 (선택) | `filter_no_entity_events=True`이면 매칭 기업 없는 이벤트 제거 |

**심각도 계산 공식:**
```
severity = clip(base + keyword_weight × keyword_count + finbert_weight × sentiment, 0, max_cap)
# 기본값: base=1.0, keyword_weight=0.4, finbert_weight=0.2, max_cap=5.0
```

**출력 컬럼:**
| 컬럼 | 타입 | 설명 |
|------|------|------|
| `event_id` | str | URL+날짜의 MD5 해시 |
| `event_time` | datetime | 이벤트 발생 시각 |
| `url` | str | 기사 URL |
| `title` | str | 기사 제목 |
| `risk_types` | str | 쉼표 구분 리스크 유형 (geopolitical, logistics, climate, other) |
| `severity` | float | 심각도 점수 [0, 5.0] |
| `entity_ids` | list | 매칭된 기업 ID 리스트 |
| `entity_scores` | list | 매칭 상세 [{company_id, score, alias}] |
| `finbert` | dict | (선택) {negative, neutral, positive} 확률 |

---

#### entity_linking.py — 엔티티 링킹

기사 제목에서 언급된 기업을 식별하는 모듈입니다.

**EntityLinker 클래스:**
```python
class EntityLinker:
    alias_to_company: Dict[str, str]  # 별칭 → company_id
    alias_list: List[str]             # 전체 별칭 목록
    min_score: int = 90               # 최소 퍼지 매칭 점수
    max_entities: int = 3             # 기사당 최대 엔티티 수
```

| 메서드 | 설명 |
|--------|------|
| `EntityLinker.from_universe(universe, ...)` | 유니버스에서 별칭 사전 구축 |
| `link_title(title)` | 제목에서 기업 매칭 → `[(company_id, score, alias), ...]` |

**매칭 전략 (2단계):**
1. **정확 매칭**: 별칭이 제목에 단어 경계로 포함되면 score=100
2. **퍼지 매칭**: 정확 매칭 실패 시 `rapidfuzz.fuzz.partial_ratio` 사용 (threshold: `min_score`)

**별칭 생성 규칙 (`build_aliases_for_company`):**
- `canonical_name` (+ 기업 접미사 제거 변형: Co., Ltd., Inc. 등)
- `company_id` (선택)
- `tickers` (";" 구분, 각각 등록)
- 모든 별칭은 소문자+공백 정규화, 2자 이상만 유효

---

#### finbert.py — FinBERT 감성 분석

```python
score_texts(texts: List[str], model_name: str = "ProsusAI/finbert") -> List[FinBertResult]
```

| 파라미터 | 설명 |
|----------|------|
| `texts` | 분석할 텍스트 리스트 |
| `model_name` | HuggingFace 모델 ID (기본: ProsusAI/finbert) |

**FinBertResult:**
```python
class FinBertResult:
    negative: float    # 부정 확률
    neutral: float     # 중립 확률
    positive: float    # 긍정 확률
```

- `transformers` + `torch` 필요 (requirements_nlp.txt)
- 모델은 LRU 캐시로 1회만 로드
- 입력 최대 256 토큰으로 자름
- `torch.no_grad()` + softmax로 확률 계산

---

### 5.4 지식 그래프 (src/kg/)

#### build_static.py — 정적 KG 구축

```python
load_universe(universe_csv: Path) -> pd.DataFrame
```
- 유니버스 CSV 로드, 필수 컬럼 누락 시 `pd.NA`로 채움
- 필수 컬럼: `company_id`, `canonical_name`, `country`, `region_group`, `value_chain_stage`, `listed`, `exchanges`, `tickers`, `notes`

```python
load_seed_edges(seed_csv: Path) -> pd.DataFrame
```
- 시드 엣지 CSV 로드
- 파일 미존재 시 빈 DataFrame 반환 (정상 동작)
- 컬럼명 정규화: `src_id` → `src_company_id`, `dst_id` → `dst_company_id`
- 필수 컬럼: `src_company_id`, `rel_type`, `dst_company_id`, `confidence_plink`, `strength`, `evidence`, `source`, `valid_from`, `valid_to`

```python
build_static_kg(
    universe: pd.DataFrame,
    seed_edges: pd.DataFrame,
    allowed_relations: list[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]
```
- **노드** = 유니버스 전체 기업
- **엣지** = `allowed_relations`에 포함된 관계만 필터
- `confidence_plink` 정규화 (숫자 변환, 기본값 0.5)
- `strength` 정규화 (숫자 변환, 기본값 1.0)
- 반환: `(nodes_df, edges_df)`

---

#### llm_extractor.py — LLM 기반 관계 추출 (실험적)

```python
class SupplyChainExtractor:
    slm: str = "Qwen2.5-3B"        # 후보 생성용 소형 모델
    llm: str = "Llama-3.1-8B"      # 검증용 대형 모델
    ontology: List[str]             # 관계 유형 목록
```

| 메서드 | 설명 |
|--------|------|
| `extract_candidates(text)` | Phase 1: SLM으로 관계 후보 추출 (현재 스텁) |
| `verify_with_reflection(candidates, context)` | Phase 2: LLM으로 검증 (confidence > 0.85 필터) |
| `run_pipeline(articles)` | E2E: 기사 → 관계 트리플릿 DataFrame |

> 현재 스텁 구현 상태. 향후 실제 LLM 연동 예정.

---

### 5.5 그래프 분석 (src/graph/)

#### baseline_exposure.py — 리스크 노출도 계산

이 모듈은 공급망 그래프에서 리스크가 어떻게 전파되는지를 계산합니다.

**그래프 구축:**
```python
build_graph(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    weight_mode: str = "confidence_times_strength"
) -> nx.DiGraph
```

| weight_mode | 엣지 가중치 |
|-------------|------------|
| `"confidence_only"` | confidence_plink |
| `"strength_only"` | strength |
| `"confidence_times_strength"` | confidence_plink × strength |

**노출도 계산 방법 1 — 최단경로:**
```python
exposure_shortest_path(G, event_nodes, severity, lam) -> Dict[str, float]
```
- 이벤트 노드에서 모든 노드까지의 최단경로 거리 계산
- 노출도 = `severity × exp(-λ × distance)`
- 거리가 멀수록 노출도 감소 (지수 감쇠)

**노출도 계산 방법 2 — Random Walk with Restart (RWR):**
```python
exposure_rwr(G, event_nodes, severity, restart_prob=0.15, iters=50) -> Dict[str, float]
```
- 이벤트 노드에서 시작하는 랜덤 워크
- 각 스텝에서 `restart_prob` 확률로 이벤트 노드로 복귀
- `iters`회 반복 후 수렴된 확률분포 × severity = 노출도
- 가중치 있는 엣지를 통한 전파 반영

**통합 함수:**
```python
compute_exposure(
    nodes, edges, risk_events,
    use_undirected: bool,    # True: 무방향 그래프로 변환
    lam: float,              # 최단경로 감쇠율 (기본 0.7)
    restart_prob: float,     # RWR 재시작 확률 (기본 0.15)
    iters: int,              # RWR 반복 횟수 (기본 50)
    weight_mode: str,        # 엣지 가중치 방식
    cfg_gnn: dict = None     # GNN 설정 (있으면 GAT도 실행)
) -> pd.DataFrame
```

**출력 컬럼:** `event_id`, `company_id`, `exposure_sp` (최단경로), `exposure_rwr` (RWR), [`exposure_gat`] (선택)

---

### 5.6 GNN 모델 (src/gnn/)

#### gat.py — Graph Attention Network

```python
class GATRiskModel(nn.Module):
    # Layer 1: GATConv(in, hidden, heads=4, dropout=0.1, edge_dim=1) + ELU
    # Dropout(0.1)
    # Layer 2: GATConv(hidden×4, out, heads=1, concat=False, dropout=0.1, edge_dim=1)
```

| 파라미터 | 기본값 | 설명 |
|----------|--------|------|
| `in_channels` | 2 | 노드 피처 차원 [리스크심각도, 밸류체인단계] |
| `hidden_channels` | 16 | 은닉 차원 |
| `out_channels` | 1 | 출력 차원 (노출도 스칼라) |

```python
run_gat_exposure(nodes, edges, risk_events, cfg_gnn) -> pd.DataFrame
```

**동작 과정:**
1. 노드 피처 구성: `[current_risk_severity, stage_index]`
2. 엣지 텐서 구성: `edge_index` + `edge_attr` (confidence × strength)
3. 각 이벤트별로:
   - 해당 기업의 리스크 심각도를 노드 피처에 주입
   - GAT forward pass → ReLU → 노출도 추출
4. 결과: `(event_id, company_id, exposure_gat)`

> `torch-geometric` 필요 (requirements_gnn.txt)

---

### 5.7 재무 분석 (src/finance/)

#### event_study.py — 이벤트 스터디

리스크 이벤트가 기업 주가에 미친 **비정상 수익률(CAR)**을 측정합니다.

```python
run_event_study(
    prices: pd.DataFrame,
    risk_events: pd.DataFrame,
    exposure: pd.DataFrame,
    windows: List[int],           # 이벤트 윈도우 (거래일). 기본: [21, 63, 126, 252]
    est_win: Tuple[int, int],     # 추정 윈도우. 기본: (-120, -20)
    topk: int = 15                # 이벤트당 상위 K개 기업
) -> pd.DataFrame
```

**분석 단계:**

1. **수익률 계산**: `compute_returns()` — 일별 로그 수익률
2. **시장 프록시**: `build_market_proxy()` — 전 기업 평균 수익률
3. **대상 기업 선정**: 이벤트별로 직접 언급 기업 + exposure_rwr 상위 K개
4. **CAR 계산**: 각 (이벤트, 기업, 윈도우)에 대해:
   - 추정 윈도우에서 OLS 회귀: `ret = α + β × mkt_ret`
   - 이벤트 윈도우에서 기대 수익률 계산
   - 비정상 수익률(AR) = 실제 수익률 - 기대 수익률
   - CAR = ΣAR (이벤트 윈도우 내 누적)

**검증 조건:**
- 추정 윈도우 내 최소 30일 데이터 필요
- 이벤트 윈도우 내 충분한 데이터 필요

**출력 컬럼:** `event_id`, `company_id`, `window_td` (거래일수), `CAR` (누적 비정상 수익률)

---

#### backtest.py — 리스크 회피 백테스트

리스크가 높은 기업을 주기적으로 제외하는 투자 전략을 백테스트합니다.

```python
backtest_exclude(
    prices: pd.DataFrame,
    exposure: pd.DataFrame,
    risk_events: pd.DataFrame,
    exclude_quantile: float = 0.2,    # 제외 비율 (상위 20%)
    decay_lambda: float = 0.03        # 리스크 감쇠율
) -> Tuple[pd.DataFrame, Dict[str, float]]
```

**전략 규칙:**
1. **일별 리스크 점수**: 각 기업의 누적 리스크 = 이전 리스크 × `exp(-decay_lambda)` + 신규 이벤트 노출도
2. **주간 리밸런싱** (금요일):
   - 리스크 상위 `(1-exclude_quantile)%` 기업 제외
   - 나머지 기업을 동일 가중치로 편입
3. **벤치마크**: 전 기업 동일 가중치 (매일)

**반환값:**
- `equity DataFrame`: 일별 벤치마크/전략 수익률 및 누적 자산곡선
- `metrics dict`: 아래 표 참조

| 지표 | 설명 |
|------|------|
| `bench_cagr` / `strat_cagr` | 연환산 수익률 |
| `bench_vol` / `strat_vol` | 연환산 변동성 |
| `bench_sharpe` / `strat_sharpe` | 샤프 비율 |
| `bench_mdd` / `strat_mdd` | 최대 낙폭 |

---

#### metrics.py — 성과 지표

```python
max_drawdown(equity: pd.Series) -> float
```
- 최대 낙폭 = `min(equity / running_max - 1)` (음수)

```python
annualized_return(equity: pd.Series, periods_per_year: int = 252) -> float
```
- 연환산 수익률 = `(최종값/최초값)^(1/연수) - 1`

```python
annualized_vol(returns: pd.Series, periods_per_year: int = 252) -> float
```
- 연환산 변동성 = `std(returns) × √252`

```python
sharpe(returns: pd.Series, rf: float = 0.0, periods_per_year: int = 252) -> float
```
- 샤프 비율 = `(평균초과수익 / 초과수익표준편차) × √252`
- 변동성 0이면 0 반환

---

### 5.8 데이터 품질 (src/quality/)

#### dq_gate.py — 통합 DQ 리포트

```python
build_dq_report(articles, prices, risk_events, nodes, edges, cfg) -> str
```

각 모듈의 QC 결과를 종합하여 마크다운 리포트를 생성합니다.

**리포트 구성:**
1. **뉴스 QC**: 중복 제거 전후 행수, 중복 비율
2. **주가 QC**: 기업수, 결측률, 이상치(일수익률 ±20% 초과) 상위 15개
3. **엔티티 QC**: 이벤트 수, 링킹 성공률, 실패 예시 10개
4. **KG QC**: 엣지 수, 근거 비율, 잘못된 관계, 누락 노드

---

#### entity_qc.py

```python
run_entity_qc(risk_events: pd.DataFrame, topn: int = 10) -> EntityQC
```
- `entity_ids`가 비어있는 이벤트 비율 측정
- 실패한 이벤트 예시 반환 (제목, URL 등)

---

#### kg_qc.py

```python
run_kg_qc(nodes: pd.DataFrame, edges: pd.DataFrame, allowed_relations: list) -> KGQC
```
- evidence 필드 비어있는 엣지 비율
- 허용되지 않은 관계 유형 체크
- 유니버스에 없는 노드를 참조하는 엣지 체크

---

#### news_qc.py

```python
run_news_qc(df: pd.DataFrame, dedup_keys: List[str]) -> Tuple[NewsQC, pd.DataFrame]
```
- 지정된 키(기본: `url`)로 중복 제거
- 중복 비율 = `(전 - 후) / 전`

---

#### prices_qc.py

```python
run_prices_qc(prices: pd.DataFrame, missing_rate_max: float = 0.03) -> PricesQC
```
- 기업별 결측률 (NaN adj_close 비율)
- 기업별 이상치율 (|일수익률| > 20%)

---

### 5.9 리포트 생성 (src/reporting/)

#### report_all.py

```python
write_md(path: Path, content: str) -> None
```
- 마크다운 파일 저장 (부모 디렉토리 자동 생성)

```python
build_index(reports_dir: Path) -> None
```
- `reports/` 내 모든 `.md` 파일을 스캔하여 `index.md` 생성

---

### 5.10 보조 도구 (src/tools/)

#### env_doctor.py — 환경 점검

```python
doctor(cfg_raw: Dict, root_dir: Path, paths: Dict) -> DoctorResult
```

| 점검 항목 | 설명 |
|-----------|------|
| 필수 파일 | `universe_csv`, `seed_edges_csv` 존재 여부 |
| FinBERT 의존성 | `torch`, `transformers` 설치 여부 (설정에서 활성화 시) |
| GNN 의존성 | `torch`, `torch_geometric` 설치 여부 (설정에서 활성화 시) |
| tabulate | 마크다운 테이블 생성용 패키지 |

**반환:** `DoctorResult(ok: bool, messages: List[str])`

---

#### seed_from_xlsx.py — Excel → 시드 엣지

```python
seed_edges_from_excel(
    xlsx_path: Path,
    universe: pd.DataFrame,
    mapping_yaml: Path,          # seed_mapping.yaml
    normalize_rules: Dict,
    allowed_relations: List[str],
    sheet: str | int | None = None,
    min_fuzzy_score: int = 88
) -> SeedImportResult
```

**처리 과정:**
1. `seed_mapping.yaml`에서 컬럼 후보명/기본값/관계 정규화 규칙 로드
2. Excel 시트 읽기
3. 컬럼 추론: 후보 리스트에서 대소문자 무시 매칭
4. 티커 별칭 맵 + EntityLinker 구축
5. 행별로:
   - src/dst → `company_id` 매핑 (직접 매칭 → 퍼지 매칭)
   - 관계 유형 정규화 (예: "공급", "납품" → SUPPLIES)
   - confidence (기본 0.6), strength (기본 1.0) 파싱
6. 양쪽 모두 매핑된 행만 출력
7. 매핑률/실패 예시 집계

**SeedImportResult:**
```python
class SeedImportResult:
    edges: pd.DataFrame          # 변환된 시드 엣지
    rows_in: int                 # 입력 행수
    rows_out: int                # 출력 행수
    mapped_src_rate: float       # src 매핑 성공률
    mapped_dst_rate: float       # dst 매핑 성공률
    dropped_rows: int            # 누락 행수
    unmapped_examples: pd.DataFrame  # 실패 예시 (디버깅용)
```

---

#### seed_from_docx.py — DOCX → 시드 엣지

```python
seed_edges_from_docx_tables(
    docx_path: Path,
    temp_xlsx_path: Path,
    *args, **kwargs
) -> SeedImportResult
```

1. `python-docx`로 DOCX 내 모든 표(Table) 추출
2. 임시 Excel 파일로 변환
3. `seed_edges_from_excel()` 호출
4. 임시 파일 정리

---

#### seed_from_folder.py — 폴더 일괄 임포트

```python
import_seed_from_folder(
    folder: Path,
    universe: pd.DataFrame,
    mapping_yaml: Path,
    normalize_rules: Dict,
    allowed_relations: List[str],
    min_fuzzy_score: int = 88
) -> FolderImportResult
```

- 폴더 내 모든 `.xlsx` / `.docx` 파일을 자동 처리
- 결과를 병합하고 `(src_company_id, rel_type, dst_company_id)` 기준 중복 제거
- 파일별 통계 + 전체 미매핑 예시 반환

---

#### universe_utils.py — 유니버스 정규화

```python
normalize_universe_df(
    universe: pd.DataFrame,
    priority_suffixes: List[str],
    normalize_rules: Dict[str, str] | None = None
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame]
```

| 반환값 | 설명 |
|--------|------|
| normalized_df | 정규화된 유니버스 |
| ticker_alias_map | 티커→company_id 전역 별칭 맵 |
| duplicates_df | 중복 company_id 행 |

**정규화 과정:**
1. 필수 컬럼 보장 (없으면 NA)
2. 기업별:
   - 세미콜론(`;`) 구분 티커 파싱
   - 거래소 우선순위로 정렬하여 primary 티커 선택
   - `company_id`가 비어있으면 primary 티커로 채움
3. `provider_ticker` 생성 (yfinance 정규화)
4. 전역 별칭 맵 구축
5. 중복 company_id 식별

**보조 함수들:**
| 함수 | 설명 |
|------|------|
| `parse_semicolon_list(x)` | 세미콜론 구분 파싱 + 중복 제거 |
| `normalize_suffix_priority(tickers, priority)` | 거래소 접미사 우선순위로 정렬 |
| `choose_primary_ticker(company_id, tickers, priority)` | 대표 티커 선택 |
| `load_priority_yaml(path)` | 우선순위 YAML 로드 |
| `build_ticker_alias_map(universe, rules)` | 티커→company_id 별칭 맵 구축 |

---

#### validators.py — 데이터 검증

```python
validate_universe(universe: pd.DataFrame) -> UniverseValidation
```
- `company_id` 결측/중복 개수 집계

```python
validate_seed_edges(seed: pd.DataFrame, allowed_relations: List[str]) -> SeedValidation
```
- 잘못된 관계 유형 및 evidence 누락 개수 집계

---

#### price_converter.py — 주가 포맷 변환

```python
convert_raw_to_parquet(csv_in: Path, univ_in: Path, parquet_out: Path) -> None
```

원시 주가 CSV를 표준 Parquet 포맷으로 변환합니다.

1. 유니버스에서 티커→company_id 매핑 구축
2. `provider_ticker_yf` 및 `tickers` 필드 활용
3. 매핑 실패 시 `.SH` ↔ `.SS` 휴리스틱 적용
4. 표준 컬럼: `company_id`, `date`, `open`, `high`, `low`, `close`, `volume`, `adj_close`

---

## 6. CLI 명령어 레퍼런스

모든 명령어는 `python -m src.cli <명령> [옵션]`으로 실행합니다.

### 파이프라인 명령어

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `init` | 필수 디렉토리 생성 | `--config` |
| `doctor` | 환경/파일/의존성 점검 | `--config` |
| `normalize-universe` | 유니버스 정규화 | `--config`, `--force` |
| `make-tickers` | yfinance 티커 매핑 생성 | `--config`, `--force` |
| `ingest-gdelt` | GDELT 뉴스 수집 | `--config`, `--force` |
| `ingest-prices` | 주가 수집 | `--config`, `--force` |
| `build-risk-events` | 리스크 이벤트 추출 | `--config` |
| `build-kg` | 지식 그래프 구축 | `--config` |
| `compute-exposure` | 리스크 노출도 계산 | `--config` |
| `event-study` | CAR 이벤트 스터디 | `--config` |
| `backtest` | 리스크 회피 백테스트 | `--config` |
| `data-quality` | 데이터 품질 리포트 | `--config` |
| `report-all` | 리포트 인덱스 생성 | `--config` |
| `pipeline` | 위 전체를 순차 실행 | `--config` |
| `smoke-test` | 목 데이터로 오프라인 테스트 | `--config` |

### 시드 임포트 명령어

| 명령어 | 설명 | 주요 옵션 |
|--------|------|-----------|
| `seed-from-xlsx` | Excel → seed_edges.csv | `--xlsx`, `--sheet`, `--out-csv`, `--mode` |
| `seed-from-docx` | DOCX → seed_edges.csv | `--docx`, `--out-csv`, `--mode` |
| `seed-from-folder` | 폴더 일괄 → seed_edges.csv | `--folder`, `--out-csv`, `--mode` |
| `validate-seed` | seed_edges 검증 | `--config` |

`--mode` 옵션: `append` (기존에 추가, 중복 제거) 또는 `overwrite` (덮어쓰기)

---

## 7. 설정 파일 상세

### configs/base.yaml

```yaml
project:
  name: "EV-Battery Supply Chain Risk MVP"
  timezone: "Asia/Seoul"
  start_date: "2026-03-06"        # 분석 시작일
  end_date: "2026-06-01"          # 분석 종료일
  root_dir: "."                   # 프로젝트 루트

paths:                            # 모든 데이터/설정 파일 경로
  universe_csv: "data/universe/univers_final.csv"
  seed_edges_csv: "data/seed/seed_edges.csv"
  gdelt_articles_csv: "data/raw/gdelt/articles.csv"
  prices_parquet: "data/processed/prices.parquet"
  risk_events_parquet: "data/processed/risk_events.parquet"
  kg_nodes_parquet: "data/processed/kg_nodes.parquet"
  kg_edges_parquet: "data/processed/kg_edges.parquet"
  exposure_parquet: "data/processed/exposure.parquet"
  car_panel_parquet: "data/processed/car_panel.parquet"
  # ... 기타 경로

schema:
  allowed_relations:              # 허용된 KG 관계 유형
    - SUPPLIES
    - BUYS_FROM
    - PARTNERS_WITH
    - COMPETES_WITH
    - SUBSIDIARY_OF
    - OWNS
    - LOCATED_IN
    - PRODUCES

gdelt:
  language: "english"
  keyword_query: '"export ban" OR "sanction" OR ...'
  max_records: 5000
  retries: 3
  retry_backoff_sec: 2.0

risk:
  keywords_yaml: "configs/risk_keywords.yaml"
  use_finbert: false              # true면 FinBERT 감성 분석 추가
  severity:
    base: 1.0
    keyword_weight: 0.4
    finbert_weight: 0.2
    max_cap: 5.0

entity_linking:
  min_fuzzy_score: 80
  max_entities_per_article: 3

exposure:
  baseline:
    lambda_dist: 0.7              # 최단경로 감쇠율
    rwr_restart_prob: 0.15        # RWR 재시작 확률
    rwr_iters: 50                 # RWR 반복 횟수

finance:
  car:
    event_windows_trading_days: [21, 63, 126, 252]   # 1M, 3M, 6M, 1Y
    estimation_window: [-120, -20]
    topk_companies_per_event: 15
  backtest:
    exclude_quantile: 0.2         # 상위 20% 리스크 기업 제외
    risk_decay_lambda: 0.03       # 일별 리스크 감쇠
```

---

### configs/risk_keywords.yaml

리스크를 3가지 유형으로 분류하는 키워드 사전입니다.

| 유형 | 키워드 예시 |
|------|-------------|
| **geopolitical** | sanction, embargo, export ban, tariff, trade war, conflict, invasion |
| **logistics** | port strike, shipping delay, logistics disruption, factory shutdown, power outage |
| **climate** | earthquake, flood, wildfire, storm, drought, landslide, tsunami |

매칭되는 키워드가 없으면 `"other"`로 분류됩니다.

---

### configs/schema.yaml

KG에서 사용 가능한 노드/관계 유형을 정의합니다.

**노드 유형:** Company, Country, Commodity, Facility

**관계 유형:**

| 관계 | 의미 | 예시 |
|------|------|------|
| SUPPLIES | A가 B에 납품 | CATL → Tesla |
| BUYS_FROM | A가 B에서 구매 | Tesla → CATL |
| PARTNERS_WITH | 파트너/합작 | BYD ↔ Toyota |
| COMPETES_WITH | 경쟁 관계 | CATL ↔ LG Energy |
| SUBSIDIARY_OF | 자회사/계열사 | 자회사 → 모회사 |
| OWNS | 소유 관계 | 모회사 → 시설 |
| LOCATED_IN | 위치 | 기업 → 국가 |
| PRODUCES | 생산 | 기업 → 원자재 |

---

### configs/seed_mapping.yaml

Excel/DOCX 임포트 시 컬럼명을 자동 인식하기 위한 매핑입니다.

```yaml
columns:
  src:        [src, source, supplier, from, 공급사, 출발]
  dst:        [dst, destination, customer, to, 수요사, 도착]
  rel_type:   [rel, relation, type, 관계, 유형]
  confidence: [confidence, conf, 신뢰도]
  strength:   [strength, str, 강도]
  evidence:   [evidence, source_url, 근거, 출처]
  # ...

defaults:
  rel_type: SUPPLIES
  confidence_plink: 0.6
  strength: 1.0
  source: xlsx_import

relation_normalization:
  SUPPLIES:      [supplies, supply, supplier, 공급, 납품]
  BUYS_FROM:     [buys, buy, purchase, 구매, 매입]
  PARTNERS_WITH: [partner, partners, jv, joint venture, 파트너, 합작]
  # ...
```

---

### configs/universe_priority.yaml

대표 티커를 선택할 때 거래소 접미사의 우선순위입니다.

```yaml
ticker_suffix_priority:
  - .HK    # 홍콩
  - .SZ    # 선전
  - .SS    # 상하이
  - .KS    # 한국 (KOSPI)
  - .KQ    # 한국 (KOSDAQ)
  - .T     # 도쿄
  - .SI    # 싱가포르
  - .L     # 런던
  # ... (총 26개 거래소)
```

---

## 8. 핵심 데이터 모델

### 유니버스 (universe)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `company_id` | str | 기업 고유 식별자 (보통 티커) |
| `canonical_name` | str | 기업 정식 명칭 |
| `country` | str | 국가 |
| `region_group` | str | 지역 그룹 |
| `value_chain_stage` | str | 밸류체인 단계 (mining, refining, cell, OEM 등) |
| `listed` | str | 상장 여부 |
| `exchanges` | str | 상장 거래소 |
| `tickers` | str | 티커 목록 (세미콜론 구분) |
| `notes` | str | 비고 |

### 시드 엣지 (seed_edges)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `src_company_id` | str | 출발 기업 ID |
| `rel_type` | str | 관계 유형 (SUPPLIES, BUYS_FROM 등) |
| `dst_company_id` | str | 도착 기업 ID |
| `confidence_plink` | float | 관계 신뢰도 [0, 1] |
| `strength` | float | 관계 강도 |
| `evidence` | str | 근거 (URL, 기사 등) |
| `source` | str | 출처 |
| `valid_from` | str | 유효 시작일 |
| `valid_to` | str | 유효 종료일 |

### 리스크 이벤트 (risk_events)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `event_id` | str | 이벤트 고유 ID (MD5) |
| `event_time` | datetime | 이벤트 시각 |
| `url` | str | 기사 URL |
| `title` | str | 기사 제목 |
| `risk_types` | str | 리스크 유형 (쉼표 구분) |
| `severity` | float | 심각도 [0, 5.0] |
| `entity_ids` | list | 관련 기업 ID 리스트 |
| `entity_scores` | list | 매칭 상세 정보 |

### 노출도 (exposure)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `event_id` | str | 이벤트 ID |
| `company_id` | str | 기업 ID |
| `exposure_sp` | float | 최단경로 기반 노출도 |
| `exposure_rwr` | float | RWR 기반 노출도 |
| `exposure_gat` | float | (선택) GAT 기반 노출도 |

### CAR 패널 (car_panel)

| 컬럼 | 타입 | 설명 |
|------|------|------|
| `event_id` | str | 이벤트 ID |
| `company_id` | str | 기업 ID |
| `window_td` | int | 이벤트 윈도우 (거래일) |
| `CAR` | float | 누적 비정상 수익률 |

---

## 9. 테스트

```bash
python -m pytest tests/
```

| 테스트 파일 | 내용 |
|------------|------|
| `test_config.py` | Config.load()로 base.yaml 파싱 및 project.name 검증 |
| `test_entity_link.py` | EntityLinker 기본 매칭 (BYD, Tesla 목 유니버스) |
| `test_md_helper.py` | df_to_markdown() 출력 검증 |
| `test_pipeline_smoke.py` | src.cli 모듈 임포트 검증 |
| `test_universe_utils.py` | normalize_universe_df()의 빈 company_id 처리 및 우선순위 정렬 |

---

## 10. CI/CD

### .github/workflows/smoke-test.yml

| 항목 | 설명 |
|------|------|
| 트리거 | PR 생성 / workflow_dispatch |
| 환경 | Ubuntu-latest, Python 3.11 |
| 단계 | 의존성 설치 → `doctor` → `smoke-test` |

스모크 테스트는 목 데이터를 생성한 뒤 `build-risk-events` → `build-kg` → `compute-exposure` → `event-study` → `backtest` → `data-quality` → `report-all`을 순차 실행하여 파이프라인 무결성을 검증합니다.

---

## 11. 외부 의존성

### 코어 (requirements.txt)

| 패키지 | 용도 |
|--------|------|
| `pandas>=2.0` | 데이터 처리 |
| `numpy>=1.24` | 수치 연산 |
| `networkx>=3.2` | 그래프 알고리즘 |
| `rapidfuzz>=3.6` | 퍼지 문자열 매칭 |
| `yfinance>=0.2` | 주가 데이터 수집 |
| `gdeltdoc>=1.5.0` | GDELT DOC API |
| `statsmodels>=0.14` | OLS 회귀 (이벤트 스터디) |
| `typer>=0.12` | CLI 프레임워크 |
| `rich>=13.0` | 터미널 출력 포매팅 |
| `pyyaml>=6.0` | YAML 설정 파싱 |
| `pyarrow>=15.0` | Parquet I/O |
| `openpyxl>=3.1` | Excel 읽기/쓰기 |
| `python-docx>=1.1` | DOCX 읽기 |
| `tabulate>=0.9` | 마크다운 테이블 |
| `tqdm>=4.66` | 진행률 표시 |
| `requests>=2.31` | HTTP 요청 |

### NLP (requirements_nlp.txt)

| 패키지 | 용도 |
|--------|------|
| `torch>=2.1` | PyTorch 딥러닝 |
| `transformers>=4.40` | HuggingFace FinBERT |

### GNN (requirements_gnn.txt)

| 패키지 | 용도 |
|--------|------|
| `torch-geometric>=2.5` | GATConv 레이어 |

---

> 이 문서는 `nabi_hyoghaw` 프로젝트의 `master` 브랜치 기준으로 작성되었습니다.
