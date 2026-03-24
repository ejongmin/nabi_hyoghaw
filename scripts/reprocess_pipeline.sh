#!/bin/bash
# ================================================================
# 나비효과 파이프라인 재처리 스크립트
#
# 수정사항 적용을 위해 기존 처리 결과를 삭제하고 전체 재실행
# - GEM Co 오매칭 수정
# - 5대 OEM 추가 (Toyota, Tesla, VW, Mercedes, Rivian)
# - risk_type 6개 카테고리 확장
# - CATL alias 강화
# - "other" 이벤트 severity 감쇠
# ================================================================

set -e  # 에러 시 중단

cd "$(dirname "$0")/.."

echo "=========================================="
echo "  나비효과 파이프라인 재처리 시작"
echo "=========================================="

# 1. 기존 checkpoint 삭제 (raw GKG는 보존)
echo ""
echo "[Step 1/6] 기존 checkpoint 삭제..."
CKPT_DIR="data/raw/gdelt/risk_events_checkpoint"
if [ -d "$CKPT_DIR" ]; then
    echo "  삭제: $CKPT_DIR/*.parquet"
    rm -f "$CKPT_DIR"/risk_gdelt_gkg_*.parquet
    rm -f "$CKPT_DIR"/_merged_risk_events.parquet
    echo "  ✅ Checkpoint 삭제 완료"
else
    echo "  Checkpoint 디렉토리 없음, 스킵"
fi

# 2. 기존 처리 결과물 삭제
echo ""
echo "[Step 2/6] 기존 처리 결과물 삭제..."
rm -f data/processed/risk_events_bq.parquet
rm -f data/processed/risk_events.parquet
rm -f data/processed/risk_events_agg.parquet
rm -f data/processed/kg_nodes.parquet
rm -f data/processed/kg_edges.parquet
rm -f data/processed/exposure_baseline.parquet
rm -f data/processed/car_panel.parquet
echo "  ✅ 처리 결과물 삭제 완료"

# 3. BigQuery parquet → risk_events (가장 오래 걸림: ~30-60분)
echo ""
echo "[Step 3/6] BigQuery GKG → risk_events 재처리 (약 30-60분 소요)..."
python -m src.cli ingest-gdelt-bq --force
echo "  ✅ risk_events 재처리 완료"

# 4. Aggregation
echo ""
echo "[Step 4/6] risk_events 집계 (약 1-2분)..."
python -m src.cli aggregate
echo "  ✅ 집계 완료"

# 5. KG + Exposure
echo ""
echo "[Step 5/6] KG 빌드 + Exposure 계산..."
python -m src.cli build-kg
python -m src.cli compute-exposure
echo "  ✅ KG + Exposure 완료"

# 6. Event Study + Backtest
echo ""
echo "[Step 6/6] Event Study (CAR) + Backtest..."
python -m src.cli event-study
python -m src.cli backtest
echo "  ✅ Event Study + Backtest 완료"

echo ""
echo "=========================================="
echo "  파이프라인 재처리 완료!"
echo "=========================================="
echo ""
echo "결과 확인:"
ls -lh data/processed/risk_events_bq.parquet 2>/dev/null || echo "  risk_events_bq.parquet: 없음"
ls -lh data/processed/risk_events_agg.parquet 2>/dev/null || echo "  risk_events_agg.parquet: 없음"
ls -lh data/processed/car_panel.parquet 2>/dev/null || echo "  car_panel.parquet: 없음"
