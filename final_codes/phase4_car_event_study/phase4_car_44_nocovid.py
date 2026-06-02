"""
Task 4: COVID 기간(2020-01~2021-12) 이벤트 제외 Phase 4 robustness check
phase4_car_44_t0m.py 와 동일하되 COVID 기간 이벤트만 필터링
"""
from pathlib import Path
import sys

# phase4_car_44_t0m.py 를 동적으로 읽어서 출력경로만 교체
SCRIPT = Path(__file__).parent / "phase4_car_44_t0m.py"
ROOT   = Path(__file__).resolve().parents[1]

src = SCRIPT.read_text(encoding="utf-8")

# 출력 경로 교체
src = src.replace(
    'car_panel_phase4_44_t0m.parquet',
    'car_panel_phase4_44_nocovid.parquet'
).replace(
    'phase4_caar_44_t0m.csv',
    'phase4_caar_44_nocovid.csv'
).replace(
    'phase4_hetero_44_t0m.csv',
    'phase4_hetero_44_nocovid.csv'
).replace(
    'phase4_contagion_44_t0m.csv',
    'phase4_contagion_44_nocovid.csv'
).replace(
    'phase4_summary_44_t0m.txt',
    'phase4_summary_44_nocovid.txt'
)

# COVID 이벤트 필터 주입: load_events 함수 이후에 필터 추가
COVID_FILTER = '''
    # ── COVID 기간(2020-01~2021-12) 이벤트 제외 (robustness)
    pre_len = len(events)
    events = [e for e in events
              if not ("2020-01" <= e.shock_month <= "2021-12")]
    log.info("COVID 제외: %d → %d건 (제거 %d건)",
             pre_len, len(events), pre_len - len(events))
'''

# "KG adjacency" 로그 다음에 필터 삽입
src = src.replace(
    '    log.info("KG adjacency:',
    COVID_FILTER + '\n    log.info("KG adjacency:'
)

# 임시 파일로 저장 후 실행
tmp = ROOT / "scripts" / "_phase4_nocovid_tmp.py"
tmp.write_text(src, encoding="utf-8")

import runpy
runpy.run_path(str(tmp), run_name="__main__")
tmp.unlink()
