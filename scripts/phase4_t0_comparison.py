"""
Phase 4 T=0 비교표
목적: T0=M과 T0=M+1 두 설계 결과를 나란히 표시
방법: reports/phase4_robustness.txt에서 두 설계 행 파싱
출력: reports/phase4_t0_comparison.txt
"""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
SRC_PATH = ROOT / "reports" / "phase4_robustness.txt"
OUT_PATH = ROOT / "reports" / "phase4_t0_comparison.txt"


def parse_t0_rows(text: str) -> list[dict]:
    """
    T0=M+1 / T0=M 행을 파싱한다.
    예: "  T0=M+1 (기존)           NEG         38    +0.0284   +1.638  0.1099"
    """
    pattern = re.compile(
        r"(T0=M\+1|T0=M)\s+\([^)]+\)\s+(NEG|POS)\s+"
        r"(\d+)\s+([+-]?\d+\.\d+)\s+([+-]?\d+\.\d+)\s+(\d+\.\d+[*]*)"
    )
    rows = []
    for m in pattern.finditer(text):
        rows.append({
            "design": m.group(1),
            "shock": m.group(2),
            "N": int(m.group(3)),
            "CAAR": float(m.group(4)),
            "t_stat": float(m.group(5)),
            "p_value": m.group(6),   # 문자열 유지 (* 포함 가능)
        })
    return rows


def main():
    src_text = SRC_PATH.read_text(encoding="utf-8")
    rows = parse_t0_rows(src_text)

    if not rows:
        raise ValueError(f"T0 행을 파싱하지 못했습니다. {SRC_PATH} 내용을 확인하세요.")

    # ── 비교표 생성 ───────────────────────────────────────────────────
    lines = []
    lines.append("=" * 68)
    lines.append("Phase 4  T=0 설계 비교표  (창: [0, +21], full quality)")
    lines.append("=" * 68)
    lines.append(f"\n원본 파일: {SRC_PATH.name}\n")

    header = f"  {'설계':<14} {'충격':>6} {'N':>5} {'CAAR':>9} {'t-stat':>8} {'p-value':>9}  {'유의성':>6}"
    lines.append(header)
    lines.append("  " + "-" * 64)

    sig_map = {"NEG": {}, "POS": {}}
    for r in rows:
        pv_str = r["p_value"]
        try:
            pv_num = float(pv_str.strip("*"))
        except ValueError:
            pv_num = 1.0
        if pv_num < 0.01:
            sig = "**"
        elif pv_num < 0.05:
            sig = "*"
        elif pv_num < 0.10:
            sig = "†"
        else:
            sig = ""
        lines.append(
            f"  {r['design']:<14} {r['shock']:>6} {r['N']:>5} "
            f"{r['CAAR']:>+9.4f} {r['t_stat']:>+8.3f} {pv_str:>9}  {sig:>6}"
        )
        sig_map[r["shock"]][r["design"]] = {"CAAR": r["CAAR"], "t": r["t_stat"], "p": pv_num}

    lines.append("\n  ** p<0.01  * p<0.05  † p<0.10\n")

    # ── 설계 간 차이 요약 ─────────────────────────────────────────────
    lines.append("■ NEG 충격 — 두 설계 차이")
    if "T0=M" in sig_map["NEG"] and "T0=M+1" in sig_map["NEG"]:
        m1 = sig_map["NEG"]["T0=M+1"]
        m0 = sig_map["NEG"]["T0=M"]
        delta_caar = m0["CAAR"] - m1["CAAR"]
        lines.append(f"  T0=M   CAAR = {m0['CAAR']:+.4f}  (t={m0['t']:+.3f}, p={m0['p']:.4f})")
        lines.append(f"  T0=M+1 CAAR = {m1['CAAR']:+.4f}  (t={m1['t']:+.3f}, p={m1['p']:.4f})")
        lines.append(f"  ΔCAAR (M - M+1) = {delta_caar:+.4f}")
        lines.append(f"  해석: T0=M 부호 반전 → 충격월 즉각 반응 vs 이월 평균회귀 차이 존재")

    lines.append("\n■ POS 충격 — 두 설계 차이")
    if "T0=M" in sig_map["POS"] and "T0=M+1" in sig_map["POS"]:
        m1 = sig_map["POS"]["T0=M+1"]
        m0 = sig_map["POS"]["T0=M"]
        delta_caar = m0["CAAR"] - m1["CAAR"]
        lines.append(f"  T0=M   CAAR = {m0['CAAR']:+.4f}  (t={m0['t']:+.3f}, p={m0['p']:.4f})")
        lines.append(f"  T0=M+1 CAAR = {m1['CAAR']:+.4f}  (t={m1['t']:+.3f}, p={m1['p']:.4f})")
        lines.append(f"  ΔCAAR (M - M+1) = {delta_caar:+.4f}")

    lines.append("\n■ 결론")
    lines.append("  - T0=M+1(기존): NEG 충격 후 양의 CAAR → 평균회귀(반전) 패턴 포착")
    lines.append("  - T0=M(대안):   NEG 충격에 음의 CAAR (†) → 충격 즉각 반응 포착")
    lines.append("  - 두 설계를 나란히 보고하면 T=0 선택의 투명성 확보 및")
    lines.append("    Contrarian 전략 논거 보강 가능")

    lines.append("\n" + "=" * 68)

    report = "\n".join(lines)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print(f"\n저장 완료: {OUT_PATH}")


if __name__ == "__main__":
    main()
