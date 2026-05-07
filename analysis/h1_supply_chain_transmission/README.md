# H1: EV 배터리 공급망 리스크 이벤트 → 주가 전파

## 가설

> **H1**: EV 배터리 공급망에서 발생한 리스크 이벤트는  
> 공급망으로 연결된 기업들의 주가에 통계적으로 유의미한  
> 비정상 수익률(Abnormal Return)을 유발한다.

## 하위 가설

- **H1a**: 리스크 이벤트는 충격 기업 자신의 주가에 즉각적 영향을 미친다
- **H1b**: 공급망 하방(Downstream) 기업으로 리스크가 전파된다
- **H1c**: 공급망 상방(Upstream) 기업으로도 전파된다
- **H1d**: 전파 효과는 공급망 거리에 따라 감쇠한다 (hop1 > hop2)
- **H1e**: 공급망 경쟁사(Peer)에게도 영향이 전달된다

---

## 핵심 결과 요약 (2026-05-07 기준)

### Phase 4-A: 월별 z-score 기반 CAR (54 이벤트)

| 가설 | 결과 | 통계 |
|------|------|------|
| H1a (직접 효과) | CAAR=+3.88% (T0=M+1), -4.01% (T0=M) | p=0.089*, p=0.053* |
| H1b (Downstream) | CAAR=-0.01%, p=0.996 | ❌ 샘플 부족 |
| H1e (Peer) | CAAR=-3.80% | **p=0.050** (Cluster SE 후에도 유지) |

### Phase 4-B: GDELT 일별 이벤트 CAR (23,328 이벤트)

| 가설 | 결과 | 통계 | 판정 |
|------|------|------|------|
| H1a | CAAR=-0.05% | p=0.474 | 개별 반응 약함 |
| **H1b (Downstream)** | **CAAR=-0.35%** | **p<0.001\*\*\*** | **✅ 확인** |
| **H1c (Upstream)** | **CAAR=-0.29%** | **p<0.001\*\*\*** | **✅ 확인** |
| **H1d (거리 감쇠)** | hop1=-0.35% > hop2=+0.06% | — | **✅ 확인** |
| H1e (Peer) | CAAR=+0.04% | p=0.500 | ❌ (일별에서 비유의) |

### Robustness

- **Cluster SE (PEER)**: 95%CI=[-8.4%, -0.4%] — 0 미포함, 유의
- **T=0 민감도**: T0=M에서 CAAR=-4.01% p=0.053*
- **임계값 z<-1.5**: CAAR=+2.53% p=0.046**
- **FF3 모형**: 방향 동일, robust

### Phase 5: GNN (Dynamic GAT)

| 버전 | IC mean | AUC |
|------|---------|-----|
| Dynamic GAT v3 + trump | +0.070 | 0.449 |
| Baseline GRU | -0.053 | 0.491 |

→ GAT이 방향 예측(IC)에서 우위, 하지만 AUC는 baseline이 높음

### 백테스트

| 전략 | CAGR | Sharpe | MDD |
|------|------|--------|-----|
| Benchmark | 31.27% | 1.189 | -27.3% |
| **Contrarian** | **32.07%** | **1.208** | **-25.0%** |
| Exclude | 30.30% | 1.154 | -32.3% |

→ 역투자(Contrarian) 전략이 벤치마크 소폭 초과

---

## 주요 파일 경로

```
data/processed/
  risk_events_classified_v9.parquet      # 11.8M 이벤트 (기본 분석 데이터)
  risk_events_classified_v10_new.parquet # finbert2 병합 (4.25M 매칭)
  tone_monthly_zscore_18_v2.parquet      # 18개 기업 월별 감성 시계열
  car_panel_phase4.parquet               # Phase 4-A CAR 패널
  car_panel_gdelt_daily.parquet          # Phase 4-B GDELT 일별 CAR

data/seed/
  seed_edges_18_internal_v3.csv          # 공급망 지식 그래프 (100 엣지)

data/universe/
  companies_18.csv                       # 최종 18개 기업

scripts/
  phase4_car_event_study.py              # Phase 4-A 메인
  phase4_gdelt_daily_car.py              # Phase 4-B 핵심
  phase4_robustness.py                   # T0/ClusterSE/임계값
  phase4_did.py                          # DID 분석
  phase4_ff3.py                          # FF3 팩터
  phase4_backtest_v2.py                  # 백테스트
  phase5_dynamic_binary.py              # GNN 베이스
  phase5_trump_run.py                   # GNN + trump tariff
```

---

## 논문 기여점 (확정)

1. **방법론**: GDELT + FinBERT 감성 분석 + 공급망 KG + CAR Event Study
2. **실증 결과**: 23,328 이벤트 기반 Downstream(-0.35%***)/Upstream(-0.29%***) 전파
3. **거리 감쇠**: hop1(-0.35%) > hop2(+0.06%) — 공급망 거리에 따른 감쇠 확인
4. **투자 함의**: Contrarian 전략으로 벤치마크 소폭 초과
5. **탐색적 분석**: Dynamic GAT (IC=+0.07) — 미래 연구 방향 제시
