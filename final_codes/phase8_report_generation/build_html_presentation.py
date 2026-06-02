"""
build_html_presentation.py
============================
고품질 HTML/CSS 발표 자료 — Figure 완전 내장 (base64)
브라우저에서 열고 ← → 화살표키로 슬라이드 이동
"""
import base64
from pathlib import Path

ROOT = Path("C:/Users/john9/nabi_hyoghaw")
FIG  = ROOT / "reports/figures"
OUT  = ROOT / "reports/나비효과_발표_20260522.html"

def b64(path):
    p = Path(path)
    if not p.exists(): return ""
    data = p.read_bytes()
    return f"data:image/png;base64,{base64.b64encode(data).decode()}"

imgs = {k: b64(FIG/v) for k, v in {
    "A1": "A1_phase4_contagion_caar.png",
    "A2": "A2_regime_conditional.png",
    "A3": "A3_neg_pos_comparison.png",
    "B1": "B1_gnn_v1_vs_v2.png",
    "B2": "B2_gnn_ablation.png",
    "B3": "B3_gnn_monthly_ic_v1.png",
    "C1": "C1_data_coverage.png",
    "C2": "C2_methodology_improvement.png",
    "D1": "D1_results_dashboard.png",
}.items()}

HTML = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>나비효과 프로젝트 — 진행상황 발표 2026.05.22</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;400;500;700;900&family=Inter:wght@300;400;500;600;700;800&display=swap');

  *{{ box-sizing:border-box; margin:0; padding:0; }}
  :root{{
    --navy:#1E3A5F; --blue:#2563EB; --lightblue:#DBEAFE;
    --orange:#F59E0B; --green:#16A34A; --red:#DC2626;
    --gray:#6B7280; --light:#F3F4F6; --white:#FFFFFF;
    --text:#1F2937;
  }}
  body{{ font-family:'Noto Sans KR','Inter',sans-serif; background:#0F172A;
        display:flex; flex-direction:column; align-items:center;
        justify-content:center; min-height:100vh; overflow:hidden; }}

  /* 슬라이드 컨테이너 */
  #deck{{ width:1280px; height:720px; position:relative; }}
  .slide{{ width:1280px; height:720px; position:absolute; top:0; left:0;
           display:none; background:white; overflow:hidden;
           border-radius:8px; box-shadow:0 25px 60px rgba(0,0,0,.5); }}
  .slide.active{{ display:flex; flex-direction:column; }}

  /* 헤더 바 */
  .header{{ background:var(--navy); color:white; padding:14px 36px 12px;
            flex-shrink:0; }}
  .header h1{{ font-size:22px; font-weight:700; letter-spacing:-.3px; }}
  .header p{{ font-size:12px; color:#93C5FD; margin-top:3px; }}

  /* 콘텐츠 영역 */
  .body{{ flex:1; padding:24px 36px 16px; overflow:hidden; }}

  /* 풀스크린 이미지 슬라이드 */
  .slide.img-full .body{{ padding:16px 20px; display:flex;
                          flex-direction:column; align-items:center; }}
  .slide.img-full img{{ max-width:100%; max-height:580px;
                        object-fit:contain; border-radius:4px; }}
  .slide.img-full .caption{{ font-size:10.5px; color:var(--gray);
                              margin-top:8px; font-style:italic; }}

  /* 타이틀 슬라이드 */
  .slide.title-slide{{ background:var(--navy); }}
  .slide.title-slide .title-wrap{{
    flex:1; display:flex; flex-direction:column; justify-content:center;
    padding:0 80px; }}
  .slide.title-slide h1{{ color:white; font-size:48px; font-weight:900;
                           line-height:1.15; }}
  .slide.title-slide h2{{ color:#93C5FD; font-size:22px; font-weight:400;
                           margin-top:16px; }}
  .slide.title-slide .meta{{ color:#64748B; font-size:14px; margin-top:32px; }}
  .accent-bar{{ width:80px; height:5px; background:var(--orange);
                border-radius:3px; margin:20px 0; }}
  .title-badges{{ display:flex; gap:16px; margin-top:28px; }}
  .badge{{ background:rgba(255,255,255,.08); border:1px solid rgba(255,255,255,.15);
           border-radius:8px; padding:12px 20px; text-align:center; }}
  .badge .val{{ color:var(--orange); font-size:22px; font-weight:800; }}
  .badge .lbl{{ color:#CBD5E1; font-size:11px; margin-top:2px; }}

  /* 2단 레이아웃 */
  .two-col{{ display:grid; grid-template-columns:1fr 1fr; gap:20px; height:100%; }}
  .two-col-6-4{{ display:grid; grid-template-columns:1.5fr 1fr; gap:20px; }}

  /* 결과 카드 */
  .result-card{{ background:var(--light); border-radius:8px; padding:16px 20px;
                 border-left:4px solid var(--blue); }}
  .result-card.green{{ border-color:var(--green); }}
  .result-card.orange{{ border-color:var(--orange); }}
  .result-card.red{{ border-color:var(--red); }}
  .result-card h3{{ font-size:13px; font-weight:700; color:var(--navy);
                    margin-bottom:8px; }}
  .result-card .big-num{{ font-size:28px; font-weight:800; color:var(--blue); }}
  .result-card.green .big-num{{ color:var(--green); }}
  .result-card.red .big-num{{ color:var(--red); }}
  .result-card p{{ font-size:11px; color:var(--gray); margin-top:4px; line-height:1.5; }}

  /* 테이블 */
  table{{ width:100%; border-collapse:collapse; font-size:12px; }}
  th{{ background:var(--navy); color:white; padding:8px 12px;
       text-align:left; font-weight:600; }}
  td{{ padding:7px 12px; border-bottom:1px solid #E5E7EB; }}
  tr:nth-child(even) td{{ background:var(--light); }}
  tr.highlight td{{ background:#EFF6FF; font-weight:700; color:var(--navy); }}
  .sig{{ color:var(--blue); font-weight:800; }}
  .bh{{ color:var(--green); font-weight:800; }}
  .ns{{ color:var(--gray); }}
  .pos{{ color:var(--green); font-weight:700; }}
  .neg-v{{ color:var(--red); font-weight:700; }}

  /* 인사이트 박스 */
  .insight{{ background:linear-gradient(135deg,#EFF6FF,#DBEAFE);
             border:1px solid #BFDBFE; border-radius:8px;
             padding:14px 18px; margin-bottom:12px; }}
  .insight h4{{ color:var(--blue); font-size:13px; font-weight:700;
                margin-bottom:6px; }}
  .insight p{{ font-size:11.5px; color:var(--text); line-height:1.6; }}

  /* 페이지 네비 */
  #nav{{ display:flex; align-items:center; gap:20px; margin-top:16px; }}
  #nav button{{ background:rgba(255,255,255,.12); border:1px solid rgba(255,255,255,.2);
                color:white; padding:8px 20px; border-radius:6px; cursor:pointer;
                font-size:14px; transition:.2s; font-family:inherit; }}
  #nav button:hover{{ background:rgba(255,255,255,.22); }}
  #nav button:disabled{{ opacity:.3; cursor:default; }}
  #progress{{ color:#94A3B8; font-size:13px; min-width:80px; text-align:center; }}
  .key-hint{{ color:#475569; font-size:11px; margin-top:8px; }}

  /* 리스트 */
  ul.clean{{ list-style:none; }}
  ul.clean li{{ font-size:12px; color:var(--text); padding:5px 0;
                padding-left:16px; position:relative; line-height:1.5; }}
  ul.clean li::before{{ content:'▸'; position:absolute; left:0;
                         color:var(--blue); }}

  /* 태그 */
  .tag{{ display:inline-block; padding:3px 10px; border-radius:20px;
          font-size:11px; font-weight:600; }}
  .tag.green{{ background:#D1FAE5; color:#065F46; }}
  .tag.red{{   background:#FEE2E2; color:#991B1B; }}
  .tag.blue{{  background:#DBEAFE; color:#1E3A5F; }}
  .tag.gray{{  background:#F3F4F6; color:#374151; }}

  /* 섹션 라벨 */
  .section-label{{ font-size:11px; font-weight:700; letter-spacing:1px;
                   text-transform:uppercase; color:var(--orange);
                   margin-bottom:8px; }}
  h2.slide-title{{ font-size:20px; font-weight:800; color:var(--navy);
                   margin-bottom:16px; }}
  .divider{{ height:2px; background:linear-gradient(90deg,var(--blue),transparent);
             margin:12px 0; border-radius:2px; }}

  /* 통계 그리드 */
  .stat-grid{{ display:grid; grid-template-columns:repeat(4,1fr); gap:12px;
               margin-bottom:16px; }}
  .stat-box{{ background:var(--light); border-radius:8px; padding:14px 16px;
              border-top:3px solid var(--blue); text-align:center; }}
  .stat-box.green{{ border-color:var(--green); }}
  .stat-box.orange{{ border-color:var(--orange); }}
  .stat-box.navy{{ border-color:var(--navy); }}
  .stat-box .sv{{ font-size:24px; font-weight:800; color:var(--blue); }}
  .stat-box.green .sv{{ color:var(--green); }}
  .stat-box.orange .sv{{ color:var(--orange); }}
  .stat-box.navy .sv{{ color:var(--navy); }}
  .stat-box .sl{{ font-size:10.5px; color:var(--gray); margin-top:4px;
                  font-weight:600; }}
  .stat-box .ss{{ font-size:10px; color:var(--gray); margin-top:2px; }}
</style>
</head>
<body>

<div id="deck">

<!-- ══ SLIDE 1: 타이틀 ══════════════════════════════════════ -->
<div class="slide title-slide active">
  <div style="width:6px;background:var(--blue);flex-shrink:0;"></div>
  <div class="title-wrap">
    <div class="section-label" style="color:var(--orange);">EV Battery Supply Chain Risk Analysis</div>
    <h1>나비효과<br><span style="font-weight:300;font-size:36px;">Butterfly Effect</span></h1>
    <h2>EV 배터리 공급망 리스크 이벤트 →<br>기업 주가 영향 실증 분석</h2>
    <div class="accent-bar"></div>
    <div class="title-badges">
      <div class="badge"><div class="val">44개</div><div class="lbl">분석 기업</div></div>
      <div class="badge"><div class="val">3.4M</div><div class="lbl">뉴스 기사</div></div>
      <div class="badge"><div class="val">140개</div><div class="lbl">공급망 엣지</div></div>
      <div class="badge"><div class="val">2017–26</div><div class="lbl">분석 기간</div></div>
    </div>
    <div class="meta">진행상황 발표 · 2026년 5월 22일 (목)</div>
  </div>
</div>

<!-- ══ SLIDE 2: 연구 질문 ══════════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>연구 질문</h1><p>왜 이 연구를 하는가?</p></div>
  <div class="body">
    <div class="insight" style="margin-bottom:16px;">
      <h4 style="font-size:15px;">💡 핵심 질문</h4>
      <p style="font-size:13px;font-weight:600;color:var(--navy);">
        "공급망 2단계 밖의 기업에도 리스크가 전파되는가? — 그리고 시장은 이를 즉각 반영하는가?"
      </p>
    </div>
    <div class="two-col" style="gap:16px;margin-top:4px;">
      <div>
        <div class="section-label">연구 배경</div>
        <ul class="clean">
          <li>EV 산업 급성장으로 공급망 상호의존성 심화</li>
          <li>리튬·코발트 등 원자재 가격 급등락 반복</li>
          <li>중국 규제·트럼프 관세로 공급망 불확실성 ↑</li>
          <li>기존 연구: 직접 거래 상대(hop1) 효과만 분석</li>
        </ul>
        <div class="divider" style="margin:14px 0;"></div>
        <div class="section-label">본 연구의 기여</div>
        <ul class="clean">
          <li><strong>2-hop 간접 경로</strong>에서의 전파 효과 최초 실증</li>
          <li>레짐(거시 환경)에 따른 <strong>방향 반전</strong> 발견</li>
          <li>GNN으로 공급망 리스크 <strong>전파 구조 학습</strong></li>
        </ul>
      </div>
      <div style="background:var(--light);border-radius:8px;padding:18px;">
        <div class="section-label">Hop 개념</div>
        <div style="text-align:center;padding:16px 0;">
          <div style="font-size:11px;color:var(--gray);margin-bottom:12px;">리스크 전파 경로</div>
          <div style="display:flex;align-items:center;justify-content:center;gap:8px;flex-wrap:wrap;">
            <div style="background:var(--red);color:white;padding:8px 12px;border-radius:6px;font-size:11px;font-weight:700;">리튬광산<br><span style="font-size:9px;">충격 발생</span></div>
            <div style="color:var(--orange);font-weight:800;font-size:18px;">→</div>
            <div style="background:var(--navy);color:white;padding:8px 12px;border-radius:6px;font-size:11px;font-weight:700;">CATL<br><span style="font-size:9px;">hop1</span></div>
            <div style="color:var(--orange);font-weight:800;font-size:18px;">→</div>
            <div style="background:var(--blue);color:white;padding:8px 12px;border-radius:6px;font-size:11px;font-weight:700;">Tesla<br><span style="font-size:9px;">hop2 ★</span></div>
          </div>
          <div style="margin-top:16px;font-size:11px;color:var(--gray);">
            hop1 효과 = 기존 연구 대상<br>
            <strong style="color:var(--blue);">hop2 효과 = 본 연구의 핵심</strong>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ SLIDE 3: 파이프라인 ══════════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>분석 파이프라인 · 5단계</h1><p>오늘은 Stage 4·5 중심 보고</p></div>
  <div class="body">
    <div style="display:flex;gap:12px;align-items:stretch;margin-bottom:16px;">
      {''.join([f'''<div style="flex:1;background:{c};border-radius:8px;padding:14px;color:white;text-align:center;">
        <div style="font-size:11px;opacity:.7;margin-bottom:6px;">Stage {i+1}</div>
        <div style="font-size:13px;font-weight:700;">{name}</div>
        <div style="font-size:10px;opacity:.8;margin-top:4px;">{sub}</div>
        {"<div style='margin-top:8px;background:rgba(255,255,255,.2);border-radius:4px;padding:3px;font-size:10px;font-weight:600;'>오늘 중점</div>" if i>=3 else ""}
      </div>{"<div style='color:#94A3B8;font-weight:800;font-size:20px;align-self:center;'>→</div>" if i<4 else ""}'''
      for i,(name,sub,c) in enumerate([
        ("뉴스 수집","GDELT 3.4M건","#374151"),
        ("감성 분석","FinBERT z-score","#374151"),
        ("충격 탐지","5th percentile","#374151"),
        ("Phase 4","CAR 이벤트 스터디","var(--blue)"),
        ("Phase 5","GNN 전파 모델","var(--blue)"),
      ])])}
    </div>
    <div class="two-col">
      <div class="result-card green">
        <h3>Phase 4 · CAR Event Study</h3>
        <div class="big-num">+1.17%</div>
        <p>MIXED hop2 CAAR [0,+21]<br>t=4.75 · p&lt;0.001 · BH FDR ✓</p>
      </div>
      <div class="result-card" style="border-color:var(--orange);">
        <h3>Phase 5 · GNN 모델</h3>
        <div class="big-num" style="color:var(--orange);">5/5</div>
        <p>Real Graph &gt; GRU (AUC 기준)<br>모든 seed에서 그래프 구조 기여 확인</p>
      </div>
    </div>
  </div>
</div>

<!-- ══ SLIDE 4: Phase 4 결과 Figure ══════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>Phase 4 · 공급망 전파 효과 (Contagion)</h1><p>★ 논문의 핵심 기여 — 관계 유형별 CAAR</p></div>
  <div class="body">
    <img src="{imgs['A1']}" alt="A1 Contagion CAAR">
    <div class="caption">Figure A1. 관계 유형별 CAAR [0,+21] — Bootstrap t-stat(B=5,000), Winsorize [2%,98%], BH FDR q=0.05</div>
  </div>
</div>

<!-- ══ SLIDE 5: 핵심 결과 테이블 ══════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>Phase 4 · 핵심 수치 정리</h1><p>1-hop 비유의, 2-hop 유의 — 정보 마찰 증거</p></div>
  <div class="body">
    <table>
      <tr><th>관계 유형</th><th>N</th><th>CAAR</th><th>t-stat</th><th>p-value</th><th>유의성</th><th>BH FDR</th></tr>
      <tr><td>DIRECT (hop0)</td><td>125</td><td class="pos">+0.58%</td><td>+0.59</td><td>0.558</td><td class="ns">n.s.</td><td>-</td></tr>
      <tr><td>UPSTREAM hop1</td><td>207</td><td class="pos">+0.12%</td><td>+0.13</td><td>0.894</td><td class="ns">n.s.</td><td>-</td></tr>
      <tr class="highlight"><td>★ UPSTREAM hop2</td><td>938</td><td class="pos">+1.19%</td><td>+2.36</td><td>0.018</td><td class="sig">**</td><td>-</td></tr>
      <tr class="highlight"><td>★ MIXED hop2</td><td>3,518</td><td class="pos">+1.17%</td><td>+4.75</td><td>&lt;0.001</td><td class="sig">***</td><td class="bh">✓</td></tr>
      <tr><td>DOWNSTREAM hop2</td><td>410</td><td class="neg-v">-0.38%</td><td>-0.63</td><td>0.532</td><td class="ns">n.s.</td><td>-</td></tr>
      <tr><td>PEER hop2</td><td>45</td><td class="pos">+5.04%</td><td>+1.82</td><td>0.076</td><td class="sig">*</td><td>-</td></tr>
    </table>
    <div style="margin-top:16px;" class="insight">
      <h4>💡 정보 마찰 (Information Friction) — 본 연구의 메인 학술 기여</h4>
      <p>시장은 <strong>직접 거래 상대(hop1)의 리스크는 즉시 반영</strong>하지만,
      <strong>2단계 간접 경로(hop2)의 리스크는 늦게 인식</strong>합니다.
      MIXED_2_2는 BH FDR 다중검정 보정 후에도 유의하여 강건성이 확인됩니다.</p>
    </div>
  </div>
</div>

<!-- ══ SLIDE 6: 레짐 조건부 ══════════════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>새로운 발견 · 레짐 조건부 Contagion</h1><p>충격기(+) vs 역풍기(−) — 방향이 체계적으로 반전</p></div>
  <div class="body">
    <img src="{imgs['A2']}" alt="A2 Regime Conditional">
    <div class="caption">Figure A2. MIXED hop2 & UPSTREAM hop2 연도별 CAAR — 레짐별 방향 반전 패턴</div>
  </div>
</div>

<!-- ══ SLIDE 7: 레짐 해석 ══════════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>레짐 조건부 분석 · 해석</h1><p>왜 COVID 제외하면 비유의가 되는가?</p></div>
  <div class="body">
    <div class="two-col-6-4" style="height:100%;">
      <div>
        <table style="margin-bottom:14px;">
          <tr><th>연도</th><th>레짐</th><th>CAAR</th><th>유의</th><th>해석</th></tr>
          <tr><td>2019</td><td>Pre-COVID</td><td class="pos">+1.44%</td><td class="sig">***</td><td>공급 충격기</td></tr>
          <tr><td>2020</td><td>COVID 충격</td><td class="pos">+3.49%</td><td class="sig">***</td><td>극단 차질</td></tr>
          <tr><td>2021</td><td>회복기</td><td class="pos">+0.73%</td><td class="ns">n.s.</td><td>-</td></tr>
          <tr><td>2022</td><td>정상화</td><td class="neg-v">-1.47%</td><td class="ns">n.s.</td><td>역방향↓</td></tr>
          <tr><td>2023</td><td>IRA/中규제</td><td class="pos">+1.13%</td><td class="sig">**</td><td>규제 충격기</td></tr>
          <tr class="highlight"><td>2024</td><td>EV수요↓+관세</td><td class="neg-v">-3.81%</td><td class="sig">***</td><td>강한 역전↓</td></tr>
          <tr><td>2025</td><td>회복 기대</td><td class="pos">+0.99%</td><td class="sig">*</td><td>모멘텀↑</td></tr>
        </table>
        <div class="insight" style="margin-top:8px;">
          <h4>COVID 제외 시 비유의인 이유</h4>
          <p>양(+)과 음(−)의 효과가 서로 상쇄 → '결과 소멸'이 아닌 '레짐별 상쇄'.<br>
          단순 pooled 분석이 아닌 <strong>레짐 조건부 분석</strong>이 올바른 설계.</p>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:10px;">
        <div class="result-card green" style="flex:1;">
          <h3>⬆ 충격기 패턴</h3>
          <p style="font-size:12px;">공급 차질 → 대체 소싱 탐색<br>→ upstream 기업 수요 증가<br>→ <strong>양(+)의 전파</strong></p>
        </div>
        <div class="result-card red" style="flex:1;">
          <h3>⬇ 역풍기 패턴</h3>
          <p style="font-size:12px;">수요 급감 + 관세<br>→ 주문 감소, 재고 해소<br>→ <strong>음(−)의 전파</strong></p>
        </div>
        <div class="result-card" style="border-color:var(--orange);flex:1;">
          <h3>📌 Bullwhip Effect</h3>
          <p style="font-size:12px;">공급망 상류일수록<br>수요 변동이 증폭<br>→ 2-hop이 1-hop보다 강함</p>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ SLIDE 8: GNN v1 vs v2 Figure ══════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>Phase 5 · GNN 모델 — v1 vs v2 솔직한 평가</h1><p>seed=42 단일 vs 5-seed 검증</p></div>
  <div class="body">
    <img src="{imgs['B1']}" alt="B1 GNN v1 vs v2">
    <div class="caption">Figure B1. GNN IC 비교 — v1(seed=42 단일) vs v2(5-seed 검증, 모두 음수)</div>
  </div>
</div>

<!-- ══ SLIDE 9: GNN 해석 ══════════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>Phase 5 · 검증된 것 vs 미검증</h1><p>그래프 구조 기여는 강건, 예측력은 미검증</p></div>
  <div class="body">
    <div class="two-col" style="gap:16px;">
      <div>
        <div class="section-label" style="color:var(--green);">✅ 검증된 결과 (강건)</div>
        <div class="result-card green">
          <h3>그래프 구조 기여</h3>
          <div class="big-num">5/5</div>
          <p>Real Graph &gt; Random Graph &gt; GRU<br>모든 seed에서 AUC 일관 우위</p>
        </div>
        <div style="margin-top:12px;font-size:11.5px;color:var(--text);line-height:1.6;">
          → 공급망 KG 구조가 모델에 유용한 정보를 제공<br>
          → "예측" 아닌 "리스크 전파 구조 학습"으로 기여 재정의
        </div>
      </div>
      <div>
        <div class="section-label" style="color:var(--red);">⚠ 미검증 (추가 연구 필요)</div>
        <div class="result-card red">
          <h3>월간 수익률 예측력</h3>
          <div class="big-num">-0.030</div>
          <p>5-seed 평균 IC — 모두 음수<br>v1 IC=+0.093은 seed=42 운</p>
        </div>
        <div style="margin-top:12px;">
          <div style="font-size:11px;color:var(--gray);margin-bottom:6px;">근본 원인:</div>
          <ul class="clean">
            <li style="font-size:11px;">표본 과소: 44기업 × 15개월 = 527 obs</li>
            <li style="font-size:11px;">테스트 기간 특수성 (EV 수요 급락)</li>
            <li style="font-size:11px;">아키텍처 민감도 (1개 변경으로 성능 급변)</li>
          </ul>
        </div>
      </div>
    </div>
  </div>
</div>

<!-- ══ SLIDE 10: GNN Ablation ══════════════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>Phase 5 · GNN Ablation — 그래프 구조 기여도 검증</h1><p>Real > Random > No-Graph : AUC 기준 5/5 시드 일관</p></div>
  <div class="body">
    <img src="{imgs['B2']}" alt="B2 GNN Ablation">
    <div class="caption">Figure B2. GNN Ablation Study — 실제 공급망 KG > 랜덤 그래프 > GRU (그래프 없음)</div>
  </div>
</div>

<!-- ══ SLIDE 11: 방법론 개선 Figure ══════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>방법론 개선 (v1 → v2)</h1><p>6가지 항목 엄밀성 강화</p></div>
  <div class="body">
    <img src="{imgs['C2']}" alt="C2 Methodology">
    <div class="caption">Figure C2. v1 → v2 방법론 개선 — Before / After / 효과</div>
  </div>
</div>

<!-- ══ SLIDE 12: 데이터 커버리지 Figure ══════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>데이터 커버리지 · 기업별 현황</h1><p>Reliable 기업-월 비율 (FinBERT 기사 ≥6건)</p></div>
  <div class="body">
    <img src="{imgs['C1']}" alt="C1 Coverage" style="max-height:560px;">
    <div class="caption">Figure C1. 기업별 데이터 커버리지 — v1(59.9%) → v2(81.0%) 개선</div>
  </div>
</div>

<!-- ══ SLIDE 13: 한계 ══════════════════════════════════════════ -->
<div class="slide">
  <div class="header"><h1>현재 한계 (솔직한 평가)</h1><p>4가지 핵심 한계</p></div>
  <div class="body">
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;">
      {''.join([f'''<div style="background:var(--light);border-radius:8px;padding:14px 16px;border-left:4px solid {c};">
        <div style="font-weight:700;color:{c};margin-bottom:6px;font-size:13px;">{title}</div>
        <ul class="clean">{"".join(f"<li style='font-size:11px;'>{item}</li>" for item in items)}</ul>
        <div style="margin-top:8px;font-size:10.5px;color:var(--blue);font-style:italic;">💡 {fix}</div>
      </div>''' for title, items, c, fix in [
        ("① 중국 소형주 데이터 공백",
         ["5개 기업 reliable=0%","GDELT 영어 편향","GNN test 20.2%가 valid=False"],
         "var(--red)", "DART API 신청 (무료)"),
        ("② 월별 집계 정보 손실",
         ["T0=월 첫 거래일로 추정","T0 설계에 따라 CAAR 부호 반전","SEC 8-K NEG 9건 검정력 부족"],
         "var(--orange)", "Bloomberg/DART로 정확한 날짜 확보"),
        ("③ GNN 표본 과소",
         ["527 valid obs → IC 표준오차≈0.04","test 기간 EV 수요 급락 특수 국면","seed 민감도 확인됨"],
         "var(--orange)", "더 긴 OOS 기간, walk-forward"),
        ("④ Phase 4 외부 타당성",
         ["COVID 제외 시 비유의 (레짐 의존)","정적 공급망 그래프 (관계 변화 미반영)","레짐 조건부 분석 필요"],
         "var(--blue)", "레짐 조건부를 메인 결과로 재편"),
      ]])}
    </div>
  </div>
</div>

<!-- ══ SLIDE 14: 종합 대시보드 ══════════════════════════════════ -->
<div class="slide img-full">
  <div class="header"><h1>종합 결과 대시보드</h1><p>핵심 수치 한 장 요약</p></div>
  <div class="body">
    <img src="{imgs['D1']}" alt="D1 Dashboard">
    <div class="caption">Figure D1. 나비효과 프로젝트 종합 결과 대시보드 — 2026.05</div>
  </div>
</div>

<!-- ══ SLIDE 15: 결론 ══════════════════════════════════════════ -->
<div class="slide" style="background:var(--navy);">
  <div style="width:6px;background:var(--blue);flex-shrink:0;"></div>
  <div style="flex:1;padding:36px 48px;display:flex;flex-direction:column;justify-content:center;">
    <div class="section-label" style="color:var(--orange);">Conclusion</div>
    <h1 style="color:white;font-size:32px;margin-bottom:20px;">결론 & 논문 기여</h1>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px;">
      <div style="background:rgba(22,163,74,.15);border:1px solid rgba(22,163,74,.4);border-radius:8px;padding:18px;">
        <div style="color:#4ADE80;font-size:12px;font-weight:700;margin-bottom:8px;">✅ 강건한 기여 (논문 메인)</div>
        <div style="color:white;font-size:14px;font-weight:700;margin-bottom:8px;">Phase 4: 공급망 2-hop 전파 효과</div>
        <div style="color:#D1FAE5;font-size:11.5px;line-height:1.6;">
          MIXED_2_2: CAAR=+1.17%, t=4.75<br>
          p&lt;0.001, BH FDR ✓<br>
          v1/v2 두 방법 모두 강건<br>
          레짐별 방향 반전 신규 발견
        </div>
      </div>
      <div style="background:rgba(245,158,11,.12);border:1px solid rgba(245,158,11,.3);border-radius:8px;padding:18px;">
        <div style="color:#FCD34D;font-size:12px;font-weight:700;margin-bottom:8px;">⚠ 조건부 기여 (탐색적)</div>
        <div style="color:white;font-size:14px;font-weight:700;margin-bottom:8px;">Phase 5: GNN 구조 학습</div>
        <div style="color:#FEF3C7;font-size:11.5px;line-height:1.6;">
          Real&gt;GRU AUC: 5/5 시드 일관<br>
          그래프 구조 기여 검증<br>
          수익률 예측력은 미검증<br>
          (seed 민감도, 표본 과소)
        </div>
      </div>
    </div>
    <div style="background:rgba(255,255,255,.07);border-radius:8px;padding:14px 18px;">
      <div style="color:var(--orange);font-size:11px;font-weight:700;margin-bottom:6px;">📌 핵심 메시지</div>
      <div style="color:#E2E8F0;font-size:12px;line-height:1.7;">
        EV 배터리 공급망에서 리스크 이벤트는 2-hop 간접 경로로 전파되며, 충격기엔 양(+), 역풍기엔 음(-)으로 방향이 달라진다.<br>
        이는 시장의 정보 처리 지연(정보 마찰)의 증거이며, 데이터 확장(DART, Bloomberg)과 장기 OOS로 더 강한 결론이 가능하다.
      </div>
    </div>
    <div style="color:#475569;font-size:12px;margin-top:20px;text-align:center;">
      감사합니다 · 데이터: GDELT GKG + SEC EDGAR + Yahoo Finance · 44개 기업 · 2017–2026
    </div>
  </div>
</div>

</div><!-- #deck -->

<!-- 네비게이션 -->
<div id="nav">
  <button id="prev" onclick="move(-1)" disabled>◀ 이전</button>
  <span id="progress">1 / 15</span>
  <button id="next" onclick="move(1)">다음 ▶</button>
</div>
<div class="key-hint">← → 화살표키로 이동 · F11 전체화면</div>

<script>
  const slides = document.querySelectorAll('.slide');
  let cur = 0;
  function move(d) {{
    slides[cur].classList.remove('active');
    cur = Math.max(0, Math.min(slides.length-1, cur+d));
    slides[cur].classList.add('active');
    document.getElementById('prev').disabled = cur===0;
    document.getElementById('next').disabled = cur===slides.length-1;
    document.getElementById('progress').textContent = (cur+1)+' / '+slides.length;
  }}
  document.addEventListener('keydown', e => {{
    if(e.key==='ArrowRight'||e.key==='ArrowDown'||e.key===' ') move(1);
    if(e.key==='ArrowLeft'||e.key==='ArrowUp') move(-1);
  }});
</script>
</body>
</html>"""

OUT.write_text(HTML, encoding="utf-8")
print(f"✅ HTML 저장: {OUT}")
print(f"   파일 크기: {OUT.stat().st_size/1024/1024:.1f} MB")
print("   브라우저로 열고 ← → 화살표키로 이동")
