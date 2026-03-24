import pandas as pd
from src.tools.universe_utils import normalize_universe_df


def test_normalize_universe_df_basic():
    uni = pd.DataFrame([
        {'company_id':'', 'canonical_name':'Test', 'tickers':'AAA.HK;AAA.SZ'},
    ])
    norm, alias, dup = normalize_universe_df(uni, priority_suffixes=['.HK','.SZ'], normalize_rules={'.SH':'.SS'})
    assert norm.loc[0,'company_id'] == 'AAA.HK'
    assert 'AAA.HK' in alias
