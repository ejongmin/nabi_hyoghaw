"""
보고서 최종 수치 기반 figure 재생성
- phase5_ablation_summary.txt 기준 (SEED=42, 18개사)
- phase4_summary_44.txt 기준 (44개사 contagion)
- seed_edges_18_internal.csv 기준 KG 서브그래프
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import pandas as pd
import networkx as nx
from pathlib import Path

OUT = Path("C:/Users/john9/nabi_hyoghaw/reports/figures")

COLORS = {
    'real':   '#2563EB',   # 파랑
    'random': '#F59E0B',   # 주황
    'gru':    '#6B7280',   # 회색
    'pos':    '#16A34A',
    'neg':    '#DC2626',
    'sig':    '#1D4ED8',
    'nsig':   '#93C5FD',
}

plt.rcParams.update({
    'font.family': 'NanumGothic',
    'axes.unicode_minus': False,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 150,
})

# ══════════════════════════════════════════════════════════════════════════════
# Figure A: Ablation Study — AUC + IC>0% 비교
# ══════════════════════════════════════════════════════════════════════════════
def fig_ablation():
    models  = ['Real Graph\nGAT', 'Random Graph\nGAT', 'No Graph\nGRU']
    auc     = [0.5281, 0.5044, 0.4594]
    ic_pos  = [67, 20, 60]          # IC>0 비율(%)
    colors  = [COLORS['real'], COLORS['random'], COLORS['gru']]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    fig.suptitle('Phase 5 Ablation Study — 모델별 성능 비교\n(SEED=42, 18개사, Test: 2024-07 ~ 2026-05, 23개월)',
                 fontsize=11, fontweight='bold', y=1.02)

    # AUC
    bars = ax1.bar(models, auc, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    ax1.axhline(0.5, color='black', linewidth=1.2, linestyle='--', label='Random baseline (0.5)')
    for bar, v in zip(bars, auc):
        ax1.text(bar.get_x() + bar.get_width()/2, v + 0.002, f'{v:.4f}',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.set_ylim(0.43, 0.56)
    ax1.set_ylabel('AUC', fontsize=11)
    ax1.set_title('방향 예측 정확도 (AUC)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.set_yticks([0.44, 0.46, 0.48, 0.50, 0.52, 0.54])

    # IC>0 비율
    bars2 = ax2.bar(models, ic_pos, color=colors, width=0.5, edgecolor='white', linewidth=1.5)
    ax2.axhline(50, color='black', linewidth=1.2, linestyle='--', label='기대 기준선 (50%)')
    for bar, v in zip(bars2, ic_pos):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 1, f'{v}%',
                 ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax2.set_ylim(0, 85)
    ax2.set_ylabel('IC > 0 비율 (%)', fontsize=11)
    ax2.set_title('IC 부호 일관성 (IC>0 비율)', fontsize=10)
    ax2.legend(fontsize=9)

    patches = [mpatches.Patch(color=c, label=m.replace('\n',' '))
               for c, m in zip(colors, models)]
    fig.legend(handles=patches, loc='lower center', ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, -0.08), frameon=False)
    fig.tight_layout()
    out = OUT / 'fig_ablation_final.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'저장: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure B: Phase 5 Monthly IC — Real GAT vs Random vs GRU
# ══════════════════════════════════════════════════════════════════════════════
def fig_monthly_ic():
    months = ['2024-09','2024-10','2024-11','2024-12',
              '2025-01','2025-02','2025-03','2025-04',
              '2025-05','2025-06','2025-07','2025-08',
              '2025-09','2025-10','2025-11']

    # phase5_ablation_summary.txt 에서 Real Graph GAT 월별 IC
    real_ic = [+0.054,+0.181,+0.507,+0.279,-0.270,+0.182,-0.029,
               +0.591,+0.238,+0.377,-0.038,-0.348,+0.515,+0.490,-0.232]

    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(months))
    bar_colors = [COLORS['real'] if v >= 0 else '#BFDBFE' for v in real_ic]
    ax.bar(x, real_ic, color=bar_colors, width=0.65, edgecolor='white')
    ax.axhline(0, color='black', linewidth=1.0)
    ax.axhline(+0.166, color=COLORS['real'], linewidth=1.5, linestyle='--',
               label=f'평균 IC = +0.166')
    ax.set_xticks(x)
    ax.set_xticklabels(months, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('IC (Information Coefficient)', fontsize=10)
    ax.set_title('Real Graph GAT 월별 IC 추이 (Test set)\nReal Graph GAT: IC>0 비율 67% (15/23개월 중 23개월 기간)',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=9)
    ax.set_ylim(-0.5, 0.75)
    pos_patch = mpatches.Patch(color=COLORS['real'], label='양의 IC (예측력 있음)')
    neg_patch = mpatches.Patch(color='#BFDBFE', label='음의 IC (역방향)')
    ax.legend(handles=[pos_patch, neg_patch], loc='upper right', fontsize=9, frameon=False)
    fig.tight_layout()
    out = OUT / 'fig_monthly_ic_final.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'저장: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure C: Contagion hop별 CAAR — 44개사 최종 수치
# ══════════════════════════════════════════════════════════════════════════════
def fig_contagion():
    data = {
        'label':  ['DIRECT\n(N=125)', 'DOWNSTREAM\nhop1\n(N=161)',
                   'UPSTREAM\nhop1\n(N=203)', 'PEER\nhop1\n(N=96)',
                   'MIXED_2\nhop2\n(N=3,356)', 'UPSTREAM_2\nhop2\n(N=898)',
                   'DOWNSTREAM_2\nhop2\n(N=384)', 'PEER_2\nhop2\n(N=44)'],
        'caar':   [+0.62, -0.43, +0.05, +0.58, +0.90, +1.17, -0.49, +5.11],
        'sig':    [False, False, False, False, True, True, False, False],
        'sig_label': ['', '', '', '', '***', '**', '', '*'],
        'hop':    [0, 1, 1, 1, 2, 2, 2, 2],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = []
    for _, row in df.iterrows():
        if row['sig']:
            colors.append(COLORS['sig'])
        elif row['caar'] >= 0:
            colors.append('#93C5FD')
        else:
            colors.append('#FCA5A5')

    x = np.arange(len(df))
    bars = ax.bar(x, df['caar'], color=colors, width=0.6, edgecolor='white', linewidth=1.2)
    ax.axhline(0, color='black', linewidth=1.0)

    for bar, row in zip(bars, df.itertuples()):
        ypos = bar.get_height() + 0.1 if bar.get_height() >= 0 else bar.get_height() - 0.3
        if row.sig_label:
            ax.text(bar.get_x() + bar.get_width()/2, ypos,
                    row.sig_label, ha='center', va='bottom', fontsize=13, fontweight='bold',
                    color=COLORS['sig'])
        caar_y = bar.get_height() + 0.45 if bar.get_height() >= 0 else bar.get_height() - 0.55
        ax.text(bar.get_x() + bar.get_width()/2, caar_y,
                f'{row.caar:+.2f}%', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df['label'], fontsize=8)
    ax.set_ylabel('CAAR (%)', fontsize=11)
    ax.set_title('공급망 hop 거리별 전파 효과 — CAAR 비교 (이벤트 창 [0,+21], 44개사)\n'
                 '*** p<0.001  ** p<0.05  * p<0.10',
                 fontsize=10, fontweight='bold')
    ax.set_ylim(-1.2, 6.5)

    # hop 구분선
    ax.axvline(0.5, color='gray', linewidth=0.8, linestyle=':')
    ax.axvline(3.5, color='gray', linewidth=0.8, linestyle=':')
    ax.text(0, -1.05, 'Direct', ha='center', fontsize=8, color='gray')
    ax.text(2, -1.05, '1-hop', ha='center', fontsize=8, color='gray')
    ax.text(5.5, -1.05, '2-hop', ha='center', fontsize=8, color='gray')

    sig_patch   = mpatches.Patch(color=COLORS['sig'],   label='통계적 유의 (p<0.05)')
    pos_patch   = mpatches.Patch(color='#93C5FD',       label='비유의, 양(+)')
    neg_patch   = mpatches.Patch(color='#FCA5A5',       label='비유의, 음(−)')
    ax.legend(handles=[sig_patch, pos_patch, neg_patch], fontsize=9, frameon=False, loc='upper left')

    fig.tight_layout()
    out = OUT / 'fig_contagion_final.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'저장: {out}')


# ══════════════════════════════════════════════════════════════════════════════
# Figure D: KG 서브그래프 네트워크 (18개사, 65 엣지)
# ══════════════════════════════════════════════════════════════════════════════
def fig_kg_subgraph():
    df = pd.read_csv('C:/Users/john9/nabi_hyoghaw/data/seed/seed_edges_18_internal.csv')
    univ = pd.read_csv('C:/Users/john9/nabi_hyoghaw/data/universe/companies_18.csv')

    stage_color = {
        'Upstream/Refining': '#92400E',
        'Materials':          '#D97706',
        'Cell/Pack':          '#1D4ED8',
        'OEM':                '#15803D',
    }
    id2stage = dict(zip(univ['company_id'], univ['stage_bucket']))
    id2name  = dict(zip(univ['company_id'], univ['canonical_name']))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        G.add_edge(row['src_company_id'], row['dst_company_id'],
                   rel=row['rel_type'])

    pos = nx.spring_layout(G, seed=7, k=2.2)

    node_colors = [stage_color.get(id2stage.get(n, ''), '#888') for n in G.nodes()]
    labels = {n: id2name.get(n, n).replace(' ', '\n') for n in G.nodes()}

    edge_styles = {
        'SUPPLIES':      ('solid',  1.8, '#374151'),
        'PARTNERS_WITH': ('dashed', 1.5, '#0369A1'),
        'COMPETES_WITH': ('dotted', 1.2, '#DC2626'),
        'BUYS_FROM':     ('dashed', 1.2, '#7C3AED'),
        'OWNS':          ('solid',  1.5, '#059669'),
    }

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor('#F9FAFB')
    fig.patch.set_facecolor('#F9FAFB')

    for rel, (style, width, color) in edge_styles.items():
        edges = [(u, v) for u, v, d in G.edges(data=True) if d['rel'] == rel]
        if edges:
            nx.draw_networkx_edges(G, pos, edgelist=edges, ax=ax,
                                   style=style, width=width,
                                   edge_color=color, alpha=0.7,
                                   arrows=True, arrowsize=12,
                                   connectionstyle='arc3,rad=0.05')

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=900, alpha=0.95)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=6.5, font_weight='bold', font_color='white')

    stage_patches = [mpatches.Patch(color=c, label=s)
                     for s, c in stage_color.items()]
    edge_patches  = [mpatches.Patch(color=c,
                                    label=f'{r} ({style})')
                     for r, (style, _, c) in edge_styles.items()]
    ax.legend(handles=stage_patches + edge_patches,
              loc='lower left', fontsize=8, frameon=True,
              framealpha=0.85, ncol=2)

    ax.set_title('EV 배터리 공급망 지식 그래프 — GAT 서브그래프\n(18개 노드, 65개 엣지 | seed_edges_18_internal.csv)',
                 fontsize=11, fontweight='bold', pad=12)
    ax.axis('off')
    fig.tight_layout()
    out = OUT / 'fig_kg_subgraph_final.png'
    fig.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f'저장: {out}')


if __name__ == '__main__':
    fig_ablation()
    fig_monthly_ic()
    fig_contagion()
    fig_kg_subgraph()
    print('모든 figure 생성 완료')
