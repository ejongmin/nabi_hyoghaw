# GitHub 공유용 시작 가이드

이 레포는 **나비효과 프로젝트의 GitHub 협업 정본**입니다.

## 처음 하는 일
1. 레포 클론
2. 가상환경 생성
3. `pip install -r requirements.txt`
4. `python -m src.cli doctor --config configs/base.yaml`
5. `python -m src.cli smoke-test --config configs/base.yaml`

## 원본 자료 업로드 위치
- 기업 목록: `data/universe/`
- 팀 조사자료(xlsx/docx): `team_sources/` *(직접 생성)*
- seed template: `data/seed/seed_edges_template.xlsx`

## 실험 전 필수 규칙
- 대표 티커는 `company_id` 하나만 사용
- 관계 타입은 `configs/schema.yaml` 밖으로 나가면 안 됨
- 변경 사항은 PR + `docs/decisions.md`에 기록
