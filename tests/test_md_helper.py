from src.common.md import df_to_markdown


def test_df_to_markdown_runs():
    import pandas as pd
    df = pd.DataFrame({'a':[1,2],'b':[3,4]})
    s = df_to_markdown(df)
    assert isinstance(s, str) and len(s) > 0
