# 나비효과 프로젝트 — 핵심 코드 모음

EV 배터리 공급망 리스크가 주가에 미치는 파급 효과 분석 프로젝트의  
**최종 보고서 작성에 사용된 핵심 코드**를 Phase별로 정리한 폴더입니다.

> 최종 보고서: `Desktop/나비효과 최종보고서.pdf`  
> 분석 유니버스: 18종목(core) / 44종목(extended), 2017–2025

---

## 폴더 구조

```
final_codes/
├── phase1_data_collection/       뉴스·가격·공시 데이터 수집
├── phase2_finbert_sentiment/     FinBERT 감성 분석
├── phase3_tone_zscore/           월별 Tone Z-score 구축
├── phase4_car_event_study/       CAR 이벤트 스터디 (Phase 4)
├── phase5_gat_gnn/               GAT 기반 GNN 리스크 전파 (Phase 5)
├── phase6_kg_supply_chain/       지식 그래프(KG) 구축
├── phase7_validation_robustness/ 검증 및 Robustness 테스트
├── phase8_report_generation/     최종 보고서·피규어 생성
└── src_core/                     공용 라이브러리 (finance, gnn, kg, nlp 등)
```

---

## Phase별 설명

### Phase 1 — 데이터 수집 (`phase1_data_collection/`)

다국어 뉴스·공시·원자재 가격 등 원천 데이터 수집 스크립트.

| 파일 | 설명 |
|------|------|
| `collect_gdelt_realtime.py` | GDELT GKG 실시간 수집 (메인 뉴스 소스) |
| `01_collect_news_44.py` | 44종목 확장 유니버스용 GDELT 수집 |
| `collect_bigkinds.py` | 한국어 뉴스 (빅카인즈) |
| `collect_chinese_news_v2.py` | 중국어 뉴스 (GDELT CN + Google News) |
| `collect_jp_de_filings.py` | 일본·독일 기업 공시 (BMW, VW, BASF 등) |
| `collect_sec_edgar.py` | 미국 SEC EDGAR 8-K 공시 |
| `collect_dart_kr.py` | 한국 DART 공시 |
| `collect_google_news_zero8.py` | Google News Zero8 보완 수집 |
| `collect_industry_rss.py` | 배터리/EV 산업 전문 RSS |
| `collect_supplement_news.py` | 보완 뉴스 수집 |
| `collect_lme_prices.py` | LME 원자재 가격 (리튬, 코발트, 니켈 등) |
| `collect_commodity_prices.py` | 원자재 가격 종합 |

---

### Phase 2 — FinBERT 감성 분석 (`phase2_finbert_sentiment/`)

수집된 뉴스에 FinBERT를 적용해 감성 점수(Positive/Negative/Neutral)를 산출.

| 파일 | 설명 |
|------|------|
| `extract_finbert_input.py` | FinBERT 입력 데이터 추출 (영문) |
| `extract_zero8_finbert_input.py` | Zero8 보완 뉴스 입력 추출 |
| `Phase_B_FinBERT_v4_multilingual.ipynb` | 다국어 FinBERT 추론 (Colab) |
| `Phase_2_8_FinBERT_v2_English.py` | FinBERT v2 영문 배치 추론 |
| `Phase_2_9_FinBERT_v2_Multilingual.py` | FinBERT v2 다국어 배치 추론 |
| `colab_finbert_inference.ipynb` | Colab GPU 추론 노트북 |
| `merge_finbert_v2_to_v10.py` | FinBERT v2 결과를 v10 이벤트 DB에 병합 |
| `merge_v9_finbert_v10.py` | v9 → v10 최종 병합 |
| `02_zscore_expanding.py` | 44종목 Expanding window Z-score |

---

### Phase 3 — Tone Z-score 구축 (`phase3_tone_zscore/`)

기업별 월별 감성 Z-score(Tone) 시계열 생성. 최종 Feature로 GNN에 투입.

| 파일 | 설명 |
|------|------|
| `rebuild_zscore_v3.py` | 18종목 Z-score v3 재구축 |
| `rebuild_zscore_v3_expanding.py` | Expanding window 방식 Z-score |
| `rebuild_seed_v3.py` | 공급망 Seed Edge v3 재구축 |
| `update_tone_2026.py` | 2026년 실시간 Tone 업데이트 |
| `build_tone_v6_all44.py` | 44종목 Tone v6 통합 빌드 |
| `03b_build_tone_all44_v2.py` | 44종목 Tone v2 (Expanding 방식) |
| `fix_2026_zscores.py` | 2026년 Z-score 보정 |

---

### Phase 4 — CAR 이벤트 스터디 (`phase4_car_event_study/`)

공급망 리스크 이벤트에 따른 누적 비정상 수익률(CAAR) 추정.  
**핵심 결과: CAAR = –1.42% (–30 ~ +5일, 44종목 No-COVID)**

| 파일 | 설명 |
|------|------|
| `event_study.py` | CAR/CAAR 계산 핵심 라이브러리 (src/finance) |
| `phase4_car_44.py` | 44종목 CAR 이벤트 스터디 메인 |
| `phase4_car_44_nocovid.py` | COVID 기간 제외 최종 버전 |
| `phase4_did.py` | Difference-in-Differences (18종목) |
| `phase4_did_44.py` | DID (44종목) |
| `phase4_ff3.py` | Fama-French 3요인 모형 |
| `phase4_robustness.py` | Robustness 테스트 (18종목) |
| `phase4_robustness_44.py` | Robustness 테스트 (44종목) |
| `phase4_gdelt_daily_car.py` | GDELT 일별 CAR 분석 |
| `backtest_bootstrap_sharpe.py` | Bootstrap Sharpe ratio 백테스트 |

---

### Phase 5 — GAT 기반 GNN (`phase5_gat_gnn/`)

Graph Attention Network(GAT)로 공급망 리스크 전파 신호를 학습,  
기업별 월별 수익률 예측 (IC, Permutation test 기반 평가).

| 파일 | 설명 |
|------|------|
| `gat.py` | GAT 모델 정의 (src/gnn) |
| `phase5_dynamic_final.py` | 18종목 Dynamic GAT 최종 버전 |
| `04_phase5_gnn_44.py` | 44종목 GNN 메인 학습·평가 |
| `phase5_ablation.py` | Ablation study (18종목) |
| `06_ablation_44.py` | Ablation study (44종목) |
| `07_ablation_multiseed.py` | Multi-seed Ablation |
| `phase5_permutation_test.py` | Permutation test (18종목) |
| `05_permutation_test_44.py` | Permutation test (44종목) |
| `phase5_backtest.py` | 포트폴리오 백테스트 |
| `08_regime_conditional.py` | 레짐 조건부 분석 |
| `10_all_regime_analysis.py` | 전체 레짐 분석 통합 |
| `11_systematic_regime_layer.py` | Systematic regime layer |
| `12_mat_upstream_deep.py` | MAT 업스트림 심층 분석 |
| `hypothesis_ab.py` | 가설 A/B 테스트 |

---

### Phase 6 — 지식 그래프 구축 (`phase6_kg_supply_chain/`)

연간 보고서(PDF) 및 GDELT에서 공급망 관계를 추출해 KG 구성.

| 파일 | 설명 |
|------|------|
| `build_static.py` | 정적 KG 구축 (src/kg) |
| `llm_extractor.py` | LLM 기반 관계 추출 |
| `extract_basf_battery_kg_v2.py` | BASF 연간 보고서 → KG |
| `extract_bmw_battery_supply.py` | BMW 배터리 공급망 추출 |
| `extract_mercedes_battery_kg.py` | Mercedes 공급망 추출 |
| `compile_bmw_battery_edges.py` | BMW 엣지 통합 |
| `filing_to_seed_edges.py` | 공시 → Seed Edge 변환 |
| `classify_risk_vectorized.py` | 리스크 이벤트 벡터화 분류 |
| `classify_v3_themes.py` | 테마 분류 v3 (Aho-Corasick) |
| `rerun_aho_corasick_v2.py` | Aho-Corasick 재분류 v2 |

---

### Phase 7 — 검증 및 Robustness (`phase7_validation_robustness/`)

Walk-forward CV, DID 검증, 레짐 IC 분석 등 결과의 신뢰성 확보.

| 파일 | 설명 |
|------|------|
| `wf_cv_10fold.py` | Walk-forward 10-Fold CV |
| `wf_cv_44.py` | 44종목 Walk-forward CV |
| `did_intensity.py` | DID 강도 분석 |
| `did_parallel_trends.py` | DID 평행 추세 검증 |
| `lookback_grid.py` | Lookback window 그리드 서치 |
| `regime_ic_analysis.py` | 레짐별 IC 분석 |
| `universe_refinement.py` | 유니버스 정제 분석 |

---

### Phase 8 — 보고서·피규어 생성 (`phase8_report_generation/`)

최종 보고서 PDF와 논문용 Figure 생성.

| 파일 | 설명 |
|------|------|
| `build_capstone_final.py` | 최종 보고서 빌드 (메인) |
| `build_capstone_template.py` | 보고서 템플릿 구성 |
| `generate_figures_final.py` | 최종 Figure 4종 생성 |
| `generate_figures_and_tables.py` | Figure + Table 통합 생성 |
| `generate_all_visuals.py` | 모든 시각화 일괄 생성 |
| `generate_c2_d1.py` | C2(방법론 개선), D1(대시보드) 피규어 |
| `build_html_presentation.py` | HTML 발표 자료 빌드 |
| `build_presenter_guide.py` | 발표자 가이드 문서 생성 |
| `llm_vs_finbert_comparison.py` | LLM vs FinBERT 성능 비교 |

---

### src_core — 공용 라이브러리 (`src_core/src/`)

프로젝트 전반에서 import해 사용하는 핵심 모듈.

| 모듈 | 주요 내용 |
|------|-----------|
| `finance/event_study.py` | CAR/CAAR 계산 |
| `finance/backtest.py` | 백테스트 엔진 |
| `gnn/gat.py` | Graph Attention Network 정의 |
| `graph/baseline_exposure.py` | 기준 노출도 계산 |
| `ingest/gdelt_bq.py` | GDELT BigQuery 수집 |
| `kg/build_static.py` | 정적 KG 구축 |
| `nlp/finbert.py` | FinBERT 래퍼 |
| `nlp/risk_scoring.py` | 리스크 점수 산출 |
| `quality/dq_gate.py` | 데이터 품질 게이트 |

---

## 실행 순서 (재현 시)

```
Phase 1 → Phase 2 → Phase 3 → Phase 6(KG) → Phase 4 → Phase 5 → Phase 7 → Phase 8
  수집       감성       Tone      공급망그래프    CAR       GAT       검증       보고서
```

> Phase 6(KG)는 Phase 4/5의 그래프 구조 입력으로 사용되므로  
> Phase 3 완료 후 Phase 4 실행 전에 먼저 수행해야 합니다.
