# 나비효과 분석 폴더 구조

각 가설(Hypothesis)별로 독립적인 하위폴더를 생성해서 분석을 관리합니다.

## 폴더 구조 규칙

```
analysis/
  h{번호}_{가설_키워드}/
    README.md         ← 가설 정의, 설계, 결과 요약
    results/          ← 수치 결과 파일 (.csv, .txt)
    figures/          ← 시각화 파일 (.png)
    scripts/          ← 이 가설 전용 스크립트 (필요 시)
```

## 가설 목록

| ID | 가설 키워드 | 핵심 질문 | 상태 |
|----|------------|---------|------|
| **H1** | `supply_chain_transmission` | EV 공급망 리스크 → 주가 전파 | ✅ 완료 |
| H2 | (다음 가설) | — | 🔜 예정 |

## 새 가설 추가 방법

```bash
mkdir -p analysis/h2_your_hypothesis/{results,figures,scripts}
# analysis/h2_your_hypothesis/README.md 작성
# 관련 스크립트는 scripts/ 또는 analysis/h2_*/scripts/ 에 배치
```

## H1 핵심 결과 한줄 요약

> GDELT 일별 이벤트 23,328건 분석 결과,  
> 공급망 하방(Downstream) 기업에서 CAAR=-0.35% (p<0.001***),  
> 상방(Upstream)에서 CAAR=-0.29% (p<0.001***),  
> 거리 감쇠(hop1 < hop2) 패턴 확인.  
> Contrarian 전략 CAGR 32.07% > Benchmark 31.27%.
