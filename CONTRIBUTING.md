# CONTRIBUTING

본 프로젝트는 회사 내부에서 사용되는 데이터 표준화 파이프라인입니다. 코드 기여 시 다음 사항을 지켜 주세요.

## 코드 스타일
- 주석과 문서는 한글을 우선합니다.
- 변수와 함수 이름은 `snake_case`를 사용하고 의미가 명확한 영문 단어를 선택합니다.
- 한글 변수명은 가급적 사용하지 않습니다.
- 함수와 클래스에는 간단한 한글 요약과 인자/반환 설명을 docstring으로 작성합니다.

## pre-commit 사용법
- 최초 한 번 `pip install pre-commit` 후 `pre-commit install` 명령으로 훅을 설정합니다.
- 커밋 시 자동으로 `ruff-format`과 `ruff` 검사가 실행됩니다.
- 전체 검사를 수동으로 실행하려면 `pre-commit run --all-files` 를 사용합니다.

문서와 테스트가 없는 코드는 병합되지 않습니다. 자세한 지침은 AGENTS.md를 참고하세요.

## 버전 관리
- 기능 추가나 수정으로 버전을 갱신할 때는 아래 파일을 모두 업데이트합니다.
  1. `README.md` 첫 헤더의 버전 표기
  2. `docs/whitepaper/index.md` 상단 주석의 버전 문구
  3. `CHANGELOG.md` 최신 항목

- 버전 규칙은 docs/whitepaper/VERSIONING.md 문서를 참고하세요.
- 위 과정을 간편하게 하려면 `tools/bump_version.py` 스크립트를 사용합니다.
  예) `python tools/bump_version.py 1.0.1`
  스크립트는 README, 백서, CHANGELOG, `pipeline/__init__.py`의 버전을 동시에 수정합니다.

pre-commit 훅에서는 해당 스크립트의 `--check` 옵션을 이용해 파일 간 버전이 일치하는지 자동 검증합니다.

## 브랜치 전략
- 기본 브랜치는 `main`과 `dev`입니다. 새로운 기능은 `dev_<모듈>` 형태의 브랜치에서 개발합니다.
- 모듈 브랜치는 기능 구현 후 Pull Request로 `dev` 브랜치에 병합합니다.
- `dev` 브랜치에서 모든 기능을 통합해 테스트한 뒤 버전과 문서를 갱신합니다.
- 최종 릴리스는 `dev`에서 `main`으로 병합하며 태그(`vX.Y.Z`)를 부여합니다.

## PR 체크리스트
- 변경 사항 요약과 테스트 결과를 PR 본문에 작성합니다.
- `CHANGELOG.md`에 항목을 추가했는지 확인합니다.
- README와 백서 문서 동기화 여부를 확인합니다.
- `pre-commit` 검사(`ruff-format`, `ruff`, 버전 동기화)가 통과해야 합니다.
