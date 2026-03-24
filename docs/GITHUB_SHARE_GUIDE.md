# GitHub에서 팀원 초대 방법

## 1. 새 레포 만들기
1. GitHub 로그인
2. 오른쪽 상단 `+` → **New repository**
3. 이름 입력 (예: `nabi-effect`)
4. Public/Private 선택
5. 빈 레포 생성

## 2. 이 폴더 업로드
### 방법 A: 웹 업로드
- 레포 생성 후 `uploading an existing file` 클릭
- 이 starter 폴더 내용을 드래그해서 업로드

### 방법 B: Git 명령어
```bash
git init
git add .
git commit -m "Initial project scaffold"
git branch -M main
git remote add origin https://github.com/<YOUR_ORG_OR_ID>/<REPO>.git
git push -u origin main
```

## 3. 팀원 초대
1. 레포 메인 화면 → **Settings**
2. 왼쪽 메뉴 → **Collaborators**
3. `Add people`
4. 팀원 GitHub 아이디 입력 후 초대

## 4. 권한 추천
- 팀원: **Write**
- 발표/정리만 하는 사람도 PR 리뷰를 위해 Write 권장
- 외부 멘토는 Read 또는 Triage

## 5. 팀원들이 처음 할 일
```bash
git clone https://github.com/<YOUR_ORG_OR_ID>/<REPO>.git
cd <REPO>
python -m venv .venv
# activate env
pip install -r requirements.txt
python -m src.cli doctor --config configs/base.yaml
python -m src.cli smoke-test --config configs/base.yaml
```

## 6. 브랜치 전략 추천
- `main`: 안정 버전
- `dev`: 통합 테스트
- 개인 작업: `feat/...`, `fix/...`

예:
- `feat/seed-import`
- `feat/risk-events`
- `fix/yfinance-ticker-map`

## 7. PR 규칙
- main 직접 push 금지
- 최소 1명 리뷰 후 merge
- doctor + smoke-test 통과 후 merge
