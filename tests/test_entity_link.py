import pandas as pd
from src.nlp.entity_linking import EntityLinker

def test_entity_linker_basic():
    uni = pd.DataFrame([
        {"company_id":"1211.HK", "canonical_name":"BYD", "tickers":"1211.HK;002594.SZ"},
        {"company_id":"TSLA", "canonical_name":"Tesla", "tickers":"TSLA"},
    ])
    linker = EntityLinker.from_universe(uni, min_score=90, max_entities=2)
    hits = linker.link_title("BYD faces sanctions; Tesla supply impacted")
    cids = [h[0] for h in hits]
    assert "1211.HK" in cids or "TSLA" in cids
