from __future__ import annotations
import typer
from pathlib import Path
import pandas as pd

from src.common.config import Config, ensure_dirs
from src.common.log import setup_logging, get_logger
from src.common.io import read_df, write_df

from src.ingest.gdelt import ingest_gdelt
from src.ingest.gdelt_bq import ingest_gdelt_bq
from src.ingest.aggregate_events import aggregate_risk_events
from src.ingest.prices import ingest_prices_from_universe
from src.ingest.tickers import build_ticker_map

from src.nlp.risk_scoring import build_risk_events

from src.quality.dq_gate import build_dq_report, run_dq_gate, DQGateError

from src.kg.build_static import load_universe, load_seed_edges, build_static_kg
from src.graph.baseline_exposure import compute_exposure
from src.finance.event_study import run_event_study
from src.finance.backtest import backtest_exclude
from src.reporting.report_all import build_index, write_md

from src.ingest.mock import make_mock_articles, make_mock_prices

from src.tools.universe_utils import normalize_universe_df, load_priority_yaml
from src.tools.seed_from_xlsx import seed_edges_from_excel
from src.tools.seed_from_docx import seed_edges_from_docx_tables
from src.tools.validators import validate_universe, validate_seed_edges
from src.tools.env_doctor import doctor
from src.tools.seed_from_folder import import_seed_from_folder



app = typer.Typer(add_completion=False)

def _load_cfg(config: str) -> Config:
    cfg = Config.load(config)
    setup_logging("INFO")
    return cfg

def _resolve_universe_path(cfg: Config) -> Path:
    """Prefer normalized universe if exists; otherwise base universe."""
    base = cfg.path("universe_csv")
    # optional normalized path
    norm_rel = cfg.get("paths","universe_normalized_csv", default=None)
    if norm_rel:
        norm = (cfg.root_dir / norm_rel).resolve()
        if norm.exists():
            return norm
    return base

@app.command("init")
def cmd_init(config: str = "configs/base.yaml"):
    """Create required folders."""
    cfg = _load_cfg(config)
    ensure_dirs(
        cfg.root_dir / "data/raw/gdelt",
        cfg.root_dir / "data/processed",
        cfg.root_dir / "reports",
        cfg.root_dir / "data/seed",
        cfg.root_dir / "data/universe",
    )
    get_logger().info("✅ directories ready")

@app.command("doctor")
def cmd_doctor(config: str = "configs/base.yaml"):
    """환경/파일/옵션 의존성 점검. 실행 전에 한 번 돌리면 좋습니다."""
    cfg = _load_cfg(config)
    res = doctor(cfg.raw, cfg.root_dir, cfg.get("paths", default={}))
    for m in res.messages:
        get_logger().info(m)
    if not res.ok:
        raise typer.Exit(code=1)


@app.command("normalize-universe")
def cmd_normalize_universe(config: str = "configs/base.yaml", force: bool = False):
    """Normalize universe: primary ticker selection, ticker list cleanup, provider_ticker creation."""
    cfg = _load_cfg(config)
    base_path = cfg.path("universe_csv")
    if not base_path.exists():
        raise typer.BadParameter(f"Missing universe_csv: {base_path}")

    out_path = (cfg.root_dir / cfg.get("paths","universe_normalized_csv")).resolve()
    if out_path.exists() and not force:
        get_logger().info(f"✅ normalized universe already exists: {out_path}")
        return

    uni = pd.read_csv(base_path)
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    prio_yaml = (cfg.root_dir / cfg.get("universe_normalization","priority_yaml", default="configs/universe_priority.yaml")).resolve()
    prio = load_priority_yaml(prio_yaml)

    norm, alias_map, dup = normalize_universe_df(uni, priority_suffixes=prio, normalize_rules=rules)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    norm.to_csv(out_path, index=False, encoding="utf-8-sig")

    # validation report
    rep_path = (cfg.root_dir / cfg.get("paths","universe_validation_report_md")).resolve()
    v = validate_universe(norm)
    md = ["# Universe validation report", ""]
    md += [f"- rows: {v.rows}", f"- missing_company_id: {v.missing_company_id}", f"- duplicated_company_id: {v.duplicated_company_id}", ""]
    if len(dup) > 0:
        md += ["## Duplicate company_id rows (preview)", "", dup.head(20).to_csv(index=False), ""]
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    rep_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    get_logger().info(f"✅ normalized universe saved: {out_path}")
    get_logger().info(f"✅ universe validation report: {rep_path}")


@app.command("make-tickers")
def cmd_make_tickers(config: str = "configs/base.yaml", force: bool = False):
    """Generate provider tickers (yfinance) from universe company_id."""
    cfg = _load_cfg(config)
    uni = load_universe(_resolve_universe_path(cfg))
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    m = build_ticker_map(uni, rules)
    out_map = cfg.path("ticker_map_parquet")
    out_txt = cfg.path("tickers_generated_txt")

    write_df(m, out_map)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(m["provider_ticker"].tolist()) + "\n", encoding="utf-8")
    get_logger().info(f"✅ ticker_map saved: {out_map}")
    get_logger().info(f"✅ tickers txt saved: {out_txt}")

@app.command("ingest-gdelt")
def cmd_ingest_gdelt(config: str = "configs/base.yaml", force: bool = False):
    cfg = _load_cfg(config)
    out = cfg.path("gdelt_articles_csv")
    df = ingest_gdelt(cfg.raw, out, force=force)
    get_logger().info(f"✅ GDELT articles: {out} (rows={len(df)})")

@app.command("ingest-gdelt-bq")
def cmd_ingest_gdelt_bq(config: str = "configs/base.yaml", force: bool = False):
    """BigQuery parquet → risk_events 변환.

    data/raw/gdelt/gdelt_gkg_*.parquet 파일을 읽어
    company_ids 추출, severity 계산, risk 분류 후
    risk_events 포맷으로 저장합니다.
    """
    cfg = _load_cfg(config)
    log = get_logger()

    uni = load_universe(_resolve_universe_path(cfg))

    # 출력 경로
    bq_risk_rel = cfg.get("paths", "gdelt_bq_risk_events_parquet",
                          default="data/processed/risk_events_bq.parquet")
    out = (cfg.root_dir / bq_risk_rel).resolve()

    import shutil
    result_path = ingest_gdelt_bq(cfg.raw, uni, out, force=force)
    log.info(f"✅ BQ risk_events: {out}")

    # risk_events_parquet에도 복사 (파이프라인 호환)
    main_risk_path = cfg.path("risk_events_parquet")
    if not main_risk_path.exists() or force:
        main_risk_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(out), str(main_risk_path))
        log.info(f"✅ Copied to pipeline path: {main_risk_path}")

@app.command("ingest-prices")
def cmd_ingest_prices(config: str = "configs/base.yaml", force: bool = False):
    cfg = _load_cfg(config)
    uni = load_universe(_resolve_universe_path(cfg))
    out = cfg.path("prices_parquet")
    import pandas as pd
    proj_start = cfg.get("project","start_date")
    proj_end = cfg.get("project","end_date")
    lookback = int(cfg.get("prices","lookback_days", default=300))
    start = (pd.to_datetime(proj_start) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
    end = proj_end
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    gcfg = cfg.get("gdelt", default={})
    retries = int(gcfg.get("retries", 3))
    backoff = float(gcfg.get("retry_backoff_sec", 2.0))
    batch_size = int(cfg.get("prices","batch_size", default=30))
    df = ingest_prices_from_universe(uni, rules, out, start, end, retries=retries, backoff=backoff, force=force, batch_size=batch_size)
    get_logger().info(f"✅ prices saved: {out} (rows={len(df)})")

@app.command("build-risk-events")
def cmd_build_risk_events(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    articles_path = cfg.path("gdelt_articles_csv")
    if not articles_path.exists():
        raise typer.BadParameter(f"Missing articles: {articles_path}. Run ingest-gdelt first (or provide the file).")

    uni = load_universe(_resolve_universe_path(cfg))
    articles = pd.read_csv(articles_path)

    # dedup
    dedup_keys = cfg.get("gdelt","dedup_on", default=["url"])
    keys = [k for k in dedup_keys if k in articles.columns]
    if keys:
        articles = articles.drop_duplicates(subset=keys).copy()

    risk = build_risk_events(
        articles=articles,
        universe=uni,
        keywords_yaml=cfg.root_dir / cfg.get("risk","keywords_yaml"),
        cfg_risk=cfg.get("risk", default={}),
        cfg_link=cfg.get("entity_linking", default={}),
    )
    out = cfg.path("risk_events_parquet")
    write_df(risk, out)
    get_logger().info(f"✅ risk_events saved: {out} (rows={len(risk)})")

@app.command("seed-from-docx")
def cmd_seed_from_docx(
    config: str = "configs/base.yaml",
    docx: str = typer.Option(..., help="입력 docx 파일 경로(.docx)"),
    out_csv: str = typer.Option("", help="출력 seed_edges.csv 경로(비우면 config.paths.seed_edges_csv)"),
    mode: str = typer.Option("", help="append|overwrite (비우면 config 기본값)"),
):
    """DOCX 표(Table) -> seed_edges.csv 자동 변환(베스트 에포트)."""
    cfg = _load_cfg(config)
    docx_path = Path(docx).expanduser().resolve()
    if not docx_path.exists():
        raise typer.BadParameter(f"docx not found: {docx_path}")

    uni = load_universe(_resolve_universe_path(cfg))
    mapping_yaml = (cfg.root_dir / cfg.get("paths","seed_mapping_yaml")).resolve()
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    allowed = cfg.get("schema","allowed_relations", default=[])

    tmp_xlsx = (cfg.root_dir / "data/processed/_docx_tables_tmp.xlsx").resolve()
    min_score = int(cfg.get("seed_import","min_fuzzy_score", default=88))
    res = seed_edges_from_docx_tables(
        docx_path=docx_path,
        temp_xlsx_path=tmp_xlsx,
        universe=uni,
        mapping_yaml=mapping_yaml,
        normalize_rules=rules,
        allowed_relations=allowed,
        sheet=0,
        min_fuzzy_score=min_score,
    )

    out = Path(out_csv).expanduser().resolve() if out_csv.strip() else cfg.path("seed_edges_csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    mode2 = mode.strip() or cfg.get("seed_import","mode_default", default="append")
    if out.exists() and mode2 == "append":
        old = pd.read_csv(out)
        merged = pd.concat([old, res.edges], ignore_index=True)
        merged = merged.drop_duplicates(subset=["src_company_id","rel_type","dst_company_id"], keep="last")
        merged.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        res.edges.to_csv(out, index=False, encoding="utf-8-sig")

    rep = (cfg.root_dir / cfg.get("paths","seed_import_report_md")).resolve()
    rep.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Seed import report (docx)", ""]
    md += [f"- input_rows: {res.rows_in}", f"- output_edges: {res.rows_out}", f"- mapped_src_rate: {res.mapped_src_rate:.3f}", f"- mapped_dst_rate: {res.mapped_dst_rate:.3f}", f"- dropped_rows(unmapped): {res.dropped_rows}", ""]
    if len(res.unmapped_examples) > 0:
        md += ["## Unmapped examples (preview)", "", res.unmapped_examples.head(10).to_csv(index=False), ""]
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")

    get_logger().info(f"✅ seed_edges saved: {out} (edges={len(res.edges)})")
    get_logger().info(f"✅ seed import report: {rep}")


@app.command("seed-from-folder")
def cmd_seed_from_folder(
    config: str = "configs/base.yaml",
    folder: str = typer.Option(..., help="엑셀/문서가 들어있는 폴더 경로"),
    out_csv: str = typer.Option("", help="출력 seed_edges.csv 경로(비우면 config.paths.seed_edges_csv)"),
    mode: str = typer.Option("", help="append|overwrite (비우면 config 기본값)"),
):
    """폴더 안의 .xlsx/.docx를 모두 읽어 seed_edges.csv로 합칩니다."""
    cfg = _load_cfg(config)
    folder_path = Path(folder).expanduser().resolve()
    uni = load_universe(_resolve_universe_path(cfg))
    mapping_yaml = (cfg.root_dir / cfg.get("paths","seed_mapping_yaml")).resolve()
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    allowed = cfg.get("schema","allowed_relations", default=[])
    min_score = int(cfg.get("seed_import","min_fuzzy_score", default=88))

    res = import_seed_from_folder(
        folder=folder_path,
        universe=uni,
        mapping_yaml=mapping_yaml,
        normalize_rules=rules,
        allowed_relations=allowed,
        min_fuzzy_score=min_score,
    )

    out = Path(out_csv).expanduser().resolve() if out_csv.strip() else cfg.path("seed_edges_csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    mode2 = mode.strip() or cfg.get("seed_import","mode_default", default="append")
    if out.exists() and mode2 == "append":
        old = pd.read_csv(out)
        merged = pd.concat([old, res.merged_edges], ignore_index=True)
        merged = merged.drop_duplicates(subset=["src_company_id","rel_type","dst_company_id"], keep="last")
        merged.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        res.merged_edges.to_csv(out, index=False, encoding="utf-8-sig")

    rep = cfg.path("reports_dir") / "seed_folder_import_report.md"
    md = ["# Seed folder import report", ""]
    md += [f"- folder: {folder_path}", f"- total_edges: {len(res.merged_edges)}", ""]
    if len(res.per_file) > 0:
        md += ["## Per-file summary", "", res.per_file.to_csv(index=False), ""]
    if len(res.unmapped_preview) > 0:
        md += ["## Unmapped preview (top 20)", "", res.unmapped_preview.to_csv(index=False), ""]
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")
    get_logger().info(f"✅ seed_edges saved: {out} (edges={len(res.merged_edges)})")
    get_logger().info(f"✅ folder import report: {rep}")


@app.command("seed-from-xlsx")
def cmd_seed_from_xlsx(
    config: str = "configs/base.yaml",
    xlsx: str = typer.Option(..., help="입력 엑셀 파일 경로(.xlsx)"),
    sheet: str = typer.Option("", help="시트명(비우면 첫 시트)"),
    out_csv: str = typer.Option("", help="출력 seed_edges.csv 경로(비우면 config.paths.seed_edges_csv)"),
    mode: str = typer.Option("", help="append|overwrite (비우면 config 기본값)"),
):
    """Excel(.xlsx) -> seed_edges.csv 자동 변환."""
    cfg = _load_cfg(config)
    xlsx_path = Path(xlsx).expanduser().resolve()
    if not xlsx_path.exists():
        raise typer.BadParameter(f"xlsx not found: {xlsx_path}")

    uni = load_universe(_resolve_universe_path(cfg))
    mapping_yaml = (cfg.root_dir / cfg.get("paths","seed_mapping_yaml")).resolve()
    rules = cfg.get("ticker_provider","normalize_rules", default={})
    allowed = cfg.get("schema","allowed_relations", default=[])

    sh = None if sheet.strip()=="" else sheet.strip()
    min_score = int(cfg.get("seed_import","min_fuzzy_score", default=88))
    res = seed_edges_from_excel(
        xlsx_path=xlsx_path,
        universe=uni,
        mapping_yaml=mapping_yaml,
        normalize_rules=rules,
        allowed_relations=allowed,
        sheet=sh,
        min_fuzzy_score=min_score,
    )

    # output path
    out = Path(out_csv).expanduser().resolve() if out_csv.strip() else cfg.path("seed_edges_csv")
    out.parent.mkdir(parents=True, exist_ok=True)

    mode2 = mode.strip() or cfg.get("seed_import","mode_default", default="append")
    if out.exists() and mode2 == "append":
        old = pd.read_csv(out)
        merged = pd.concat([old, res.edges], ignore_index=True)
        merged = merged.drop_duplicates(subset=["src_company_id","rel_type","dst_company_id"], keep="last")
        merged.to_csv(out, index=False, encoding="utf-8-sig")
    else:
        res.edges.to_csv(out, index=False, encoding="utf-8-sig")

    # report
    rep = (cfg.root_dir / cfg.get("paths","seed_import_report_md")).resolve()
    rep.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Seed import report (xlsx)", ""]
    md += [f"- input_rows: {res.rows_in}", f"- output_edges: {res.rows_out}", f"- mapped_src_rate: {res.mapped_src_rate:.3f}", f"- mapped_dst_rate: {res.mapped_dst_rate:.3f}", f"- dropped_rows(unmapped): {res.dropped_rows}", ""]
    if len(res.unmapped_examples) > 0:
        md += ["## Unmapped examples (preview)", "", res.unmapped_examples.head(10).to_csv(index=False), ""]
    rep.write_text("\n".join(md) + "\n", encoding="utf-8")

    get_logger().info(f"✅ seed_edges saved: {out} (edges={len(res.edges)})")
    get_logger().info(f"✅ seed import report: {rep}")


@app.command("validate-seed")
def cmd_validate_seed(config: str = "configs/base.yaml"):
    """Validate seed_edges.csv against schema and evidence completeness."""
    cfg = _load_cfg(config)
    seed_path = cfg.path("seed_edges_csv")
    if not seed_path.exists():
        raise typer.BadParameter(f"seed_edges not found: {seed_path}")
    seed = pd.read_csv(seed_path)
    allowed = cfg.get("schema","allowed_relations", default=[])
    v = validate_seed_edges(seed, allowed_relations=allowed)
    rep_dir = cfg.path("reports_dir")
    md = ["# Seed validation report", ""]
    md += [f"- rows: {v.rows}", f"- bad_relation_rows: {v.bad_relation_rows}", f"- missing_evidence_rows: {v.missing_evidence_rows}", ""]
    out = rep_dir / "seed_validation_report.md"
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    get_logger().info(f"✅ seed validation report: {out}")


@app.command("build-kg")
def cmd_build_kg(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    uni = load_universe(_resolve_universe_path(cfg))
    seed = load_seed_edges(cfg.path("seed_edges_csv"))
    allowed = cfg.get("schema","allowed_relations", default=[])
    nodes, edges = build_static_kg(uni, seed, allowed_relations=allowed)

    write_df(nodes, cfg.path("kg_nodes_parquet"))
    write_df(edges, cfg.path("kg_edges_parquet"))
    get_logger().info(f"✅ kg saved: nodes={len(nodes)}, edges={len(edges)}")

@app.command("compute-exposure")
def cmd_compute_exposure(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    risk_path = cfg.path("risk_events_parquet")
    if not risk_path.exists():
        raise typer.BadParameter("risk_events not found. Run build-risk-events first.")
    nodes_path = cfg.path("kg_nodes_parquet")
    edges_path = cfg.path("kg_edges_parquet")
    if not nodes_path.exists() or not edges_path.exists():
        raise typer.BadParameter("kg not found. Run build-kg first.")

    risk = read_df(risk_path)
    nodes = read_df(nodes_path)
    edges = read_df(edges_path)

    use_undirected = bool(cfg.get("kg","use_undirected_for_exposure", default=True))
    lam = float(cfg.get("exposure","baseline","lambda_dist", default=0.7))
    rp = float(cfg.get("exposure","baseline","rwr_restart_prob", default=0.15))
    iters = int(cfg.get("exposure","baseline","rwr_iters", default=50))
    weight_mode = str(cfg.get("kg","edge_weight","weight_mode", default="confidence_times_strength"))

    exp = compute_exposure(nodes, edges, risk, use_undirected, lam, rp, iters, weight_mode)
    out = cfg.path("exposure_parquet")
    write_df(exp, out)
    get_logger().info(f"✅ exposure saved: {out} (rows={len(exp)})")

@app.command("event-study")
def cmd_event_study(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    prices_path = cfg.path("prices_parquet")
    risk_path = cfg.path("risk_events_parquet")
    exp_path = cfg.path("exposure_parquet")
    if not prices_path.exists():
        raise typer.BadParameter("prices not found. Run ingest-prices first.")
    if not risk_path.exists():
        raise typer.BadParameter("risk_events not found. Run build-risk-events first.")
    if not exp_path.exists():
        raise typer.BadParameter("exposure not found. Run compute-exposure first.")

    prices = read_df(prices_path)
    risk = read_df(risk_path)
    exp = read_df(exp_path)

    windows = cfg.get("finance","car","event_windows_trading_days", default=[21,63,126,252])
    est_win = tuple(cfg.get("finance","car","estimation_window", default=[-120,-20]))
    topk = int(cfg.get("finance","car","topk_companies_per_event", default=15))

    car = run_event_study(prices, risk, exp, windows=windows, est_win=est_win, topk=topk)
    out = cfg.path("car_panel_parquet")
    write_df(car, out)
    get_logger().info(f"✅ car_panel saved: {out} (rows={len(car)})")

@app.command("backtest")
def cmd_backtest(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    prices_path = cfg.path("prices_parquet")
    risk_path = cfg.path("risk_events_parquet")
    exp_path = cfg.path("exposure_parquet")
    if not prices_path.exists():
        raise typer.BadParameter("prices not found.")
    if not risk_path.exists():
        raise typer.BadParameter("risk_events not found.")
    if not exp_path.exists():
        raise typer.BadParameter("exposure not found.")

    prices = read_df(prices_path)
    risk = read_df(risk_path)
    exp = read_df(exp_path)

    q = float(cfg.get("finance","backtest","exclude_quantile", default=0.2))
    lam = float(cfg.get("finance","backtest","risk_decay_lambda", default=0.03))
    equity, metrics = backtest_exclude(prices, exp, risk, exclude_quantile=q, decay_lambda=lam)

    out_csv = cfg.root_dir / cfg.get("paths","backtest_equity_csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    equity.to_csv(out_csv, index=False, encoding="utf-8-sig")
    # metrics md
    rep_dir = cfg.path("reports_dir")
    md = ["# Backtest summary", ""]
    for k,v in metrics.items():
        md.append(f"- {k}: {v:.4f}")
    write_md(rep_dir / "backtest_summary.md", "\n".join(md) + "\n")

    get_logger().info(f"✅ backtest saved: {out_csv}")
    get_logger().info(f"✅ metrics saved: {rep_dir / 'backtest_summary.md'}")

@app.command("data-quality")
def cmd_data_quality(config: str = "configs/base.yaml", strict: bool = True):
    """DQ Gate: 데이터 품질 검사. strict=True이면 threshold 미달 시 파이프라인 중단."""
    cfg = _load_cfg(config)
    rep_dir = cfg.path("reports_dir")
    log = get_logger()

    # load available data
    articles = None
    prices = None
    risk = None
    nodes = None
    edges = None

    gdelt_source = cfg.get("gdelt", "source", default="doc_api")
    if gdelt_source == "bigquery":
        # BQ 모드: risk_events에 이미 articles 정보 포함
        articles = None  # BQ 모드에서는 별도 articles 불필요
    elif cfg.path("gdelt_articles_csv").exists():
        articles = pd.read_csv(cfg.path("gdelt_articles_csv"))
    if cfg.path("prices_parquet").exists():
        prices = read_df(cfg.path("prices_parquet"))
    if cfg.path("risk_events_parquet").exists():
        risk = read_df(cfg.path("risk_events_parquet"))
    if cfg.path("kg_nodes_parquet").exists():
        nodes = read_df(cfg.path("kg_nodes_parquet"))
    if cfg.path("kg_edges_parquet").exists():
        edges = read_df(cfg.path("kg_edges_parquet"))

    try:
        result = run_dq_gate(articles, prices, risk, nodes, edges, cfg.raw, strict=strict)
        write_md(rep_dir / "data_quality_report.md", result.report_md)
        log.info(f"✅ DQ Gate PASSED — wrote {rep_dir / 'data_quality_report.md'}")
        if result.warnings:
            for w in result.warnings:
                log.warning(f"  ⚠️ {w}")
    except DQGateError as e:
        # 리포트는 항상 저장 (디버깅용)
        write_md(rep_dir / "data_quality_report.md", e.report)
        log.error(f"❌ DQ Gate FAILED — report saved to {rep_dir / 'data_quality_report.md'}")
        for f in e.failures:
            log.error(f"  ✗ {f}")
        raise typer.Exit(code=2)

@app.command("report-all")
def cmd_report_all(config: str = "configs/base.yaml"):
    cfg = _load_cfg(config)
    build_index(cfg.path("reports_dir"))
    get_logger().info(f"✅ report index: {cfg.path('reports_dir') / 'index.md'}")

@app.command("pipeline")
def cmd_pipeline(config: str = "configs/base.yaml", skip_gate: bool = False):
    """E2E 파이프라인 실행.

    DQ Gate가 중간에 강제 삽입됩니다:
      ingest → risk_events → build_kg → ★ DQ Gate ★ → exposure → CAR → backtest
    threshold 미달 시 파이프라인이 중단됩니다.
    --skip-gate 옵션으로 DQ Gate를 경고 모드로 전환 가능.
    """
    cfg = _load_cfg(config)
    log = get_logger()

    # Phase 1: 데이터 수집 + 가공
    cmd_init(config)
    cmd_doctor(config)
    cmd_normalize_universe(config)
    cmd_make_tickers(config)

    # GDELT 소스 분기: bigquery parquet vs DOC API
    gdelt_source = cfg.get("gdelt", "source", default="doc_api")
    if gdelt_source == "bigquery":
        log.info("📡 GDELT source: BigQuery parquets")
        cmd_ingest_gdelt_bq(config)
        # BQ 경로는 ingest-gdelt-bq에서 risk_events까지 한번에 처리
        # → build-risk-events 스킵
    else:
        log.info("📡 GDELT source: DOC API")
        cmd_ingest_gdelt(config)
        cmd_build_risk_events(config)

    cmd_ingest_prices(config)

    # ★ 이벤트 집계: 400만건 → 날짜+기업 단위 (compute-exposure 메모리 안전)
    if gdelt_source == "bigquery":
        raw_risk = cfg.path("risk_events_parquet")
        agg_out = cfg.root_dir / "data/processed/risk_events_agg.parquet"
        log.info("📊 Aggregating risk events (date × entity)...")
        aggregate_risk_events(raw_risk, agg_out, force=True)
        # 집계된 파일을 파이프라인에서 사용
        import shutil
        shutil.copy2(str(agg_out), str(raw_risk))
        log.info(f"✅ risk_events.parquet replaced with aggregated version")

    cmd_build_kg(config)

    # ★ DQ Gate: red flag 시 여기서 중단 ★
    log.info("=" * 50)
    log.info("★ DQ Gate checkpoint ★")
    log.info("=" * 50)
    cmd_data_quality(config, strict=(not skip_gate))

    # Phase 2: 분석 (DQ Gate 통과 후에만 실행)
    cmd_compute_exposure(config)
    cmd_event_study(config)
    cmd_backtest(config)
    cmd_report_all(config)
    log.info("🏁 pipeline complete")

@app.command("smoke-test")
def cmd_smoke_test(config: str = "configs/base.yaml"):
    """Offline smoke test: generate mock articles/prices and run core steps."""
    cfg = _load_cfg(config)
    cmd_init(config)
    cmd_doctor(config)

    uni = load_universe(_resolve_universe_path(cfg))
    # mock data
    articles_csv = cfg.path("gdelt_articles_csv")
    prices_parquet = cfg.path("prices_parquet")
    make_mock_articles(articles_csv)
    import pandas as pd
    proj_start = cfg.get("project","start_date")
    proj_end = cfg.get("project","end_date")
    lookback = int(cfg.get("prices","lookback_days", default=300))
    start_mock = (pd.to_datetime(proj_start) - pd.Timedelta(days=lookback)).strftime("%Y-%m-%d")
    make_mock_prices(uni["company_id"].astype(str).tolist(), prices_parquet, start_mock, proj_end)

    # pipeline core
    cmd_build_risk_events(config)
    cmd_build_kg(config)
    cmd_compute_exposure(config)
    cmd_event_study(config)
    cmd_backtest(config)
    cmd_data_quality(config)
    cmd_report_all(config)
    get_logger().info("✅ smoke-test complete")

if __name__ == "__main__":
    app()
