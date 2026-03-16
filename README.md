# 나비효과 EV-배터리 공급망 리스크 MVP (v3: 안정성 강화)

이 템플릿은 **오류가 자주 나는 구간(경로/데이터 누락/티커 포맷/옵션 의존성)**을 최대한 방어적으로 처리하도록 작성되었습니다.

> 목표: 뉴스(GDELT) → 리스크 이벤트 → 공급망 그래프(KG) → 노출도(Exposure) → CAR(1M/3M/6M/1Y) → 백테스트 → 리포트

---

## 0) 전제
- Python 3.10+ 권장
- 로컬/Colab 모두 지원
- 인터넷이 안 되는 환경에서는 `mock-data`로 스모크 테스트 가능

---

## 1) 설치(로컬)
```bash
python -m pip install -r requirements.txt

# (선택) FinBERT 감성 점수까지 쓰려면
python -m pip install -r requirements_nlp.txt

# (선택) GAT까지 돌리려면
python -m pip install -r requirements_gnn.txt
```

---

## 2) 빠른 실행
```bash
python -m src.cli init --config configs/base.yaml
python -m src.cli pipeline --config configs/base.yaml
```

---

## 3) Colab 실행
1) Drive 마운트
2) Drive 안 폴더로 이동
3) `configs/base_colab_drive.yaml`의 root_dir을 본인 Drive 경로로 수정
4) 실행

---

## 4) 가장 흔한 오류/실수 방지 포인트
### (1) 티커 포맷
- 유니버스에는 `.SH`가 섞여있는데, Yahoo Finance는 보통 `.SS`를 씁니다.
- 이 템플릿은 `make-tickers` 단계에서 `.SH -> .SS`로 변환한 **yfinance 전용 티커 파일**을 생성합니다.

```bash
python -m src.cli make-tickers --config configs/base.yaml
```

### (2) optional dependency
- FinBERT/torch가 없는데 use_finbert=True면 에러가 납니다.
- GAT/torch-geometric이 없는데 gnn.enabled=True면 에러가 납니다.
- config에서 옵션을 켠 뒤, requirements_nlp/gnn을 설치하세요.

### (3) 데이터가 이미 있으면 재수집 스킵
- ingest 단계는 기본적으로 출력 파일이 이미 있으면 스킵합니다.
- 강제로 다시 받고 싶으면 `--force` 옵션을 사용하세요.

---

## 5) 스모크 테스트(인터넷 없이도 실행 가능)
```bash
python -m src.cli smoke-test --config configs/base.yaml
```


---

## (v4) 유니버스/seed 자동화

### 유니버스 정규화(대표 티커 고정/중복 점검)
```bash
python -m src.cli normalize-universe --config configs/base.yaml
```

### 엑셀 → seed_edges.csv 자동 생성
```bash
python -m src.cli seed-from-xlsx --config configs/base.yaml --xlsx /path/to/file.xlsx --sheet "Sheet1"
python -m src.cli validate-seed --config configs/base.yaml
```

### DOCX 표 → seed_edges.csv 자동 생성(베스트에포트)
```bash
python -m src.cli seed-from-docx --config configs/base.yaml --docx /path/to/file.docx
```


---

## (v4.1) 추가 기능

### 실행 전 점검(doctor)
```bash
python -m src.cli doctor --config configs/base.yaml
```

### 폴더 단위 seed import
```bash
python -m src.cli seed-from-folder --config configs/base.yaml --folder /path/to/folder
```

추가 문서: `docs/USAGE_KR.md`



## GitHub 공유 시작
- 프로젝트 시작 전: `docs/START_HERE.md`
- 팀 운영 규칙: `docs/TEAM_WORKFLOW.md`
- GitHub 공유/초대 방법: `docs/GITHUB_SHARE_GUIDE.md`
- 결정 기록: `docs/decisions.md`

## 추천 첫 실행
```bash
python -m src.cli doctor --config configs/base.yaml
python -m src.cli smoke-test --config configs/base.yaml
```
