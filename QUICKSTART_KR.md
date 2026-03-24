# 무엇을 바로 쓰면 되나요?

## 핵심
- 이 폴더를 그대로 GitHub 새 레포에 올리면 됩니다.
- 팀원 원본 자료(xlsx/docx)는 추후 `team_sources/` 폴더를 만들어 넣고,
  아래 명령으로 seed_edges를 자동 생성하세요.

```bash
python -m src.cli normalize-universe --config configs/base.yaml
python -m src.cli seed-from-folder --config configs/base.yaml --folder ./team_sources
python -m src.cli validate-seed --config configs/base.yaml
python -m src.cli pipeline --config configs/base.yaml
```

## 꼭 먼저 읽을 문서
1. `docs/START_HERE.md`
2. `docs/GITHUB_SHARE_GUIDE.md`
3. `docs/TEAM_WORKFLOW.md`
