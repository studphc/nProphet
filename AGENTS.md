# AGENTS.md

## 브랜치 및 문서 관리
- `main` → `dev` → 모듈별 `dev_모듈명` 구조로 개발한다.
- 모듈 브랜치에서는 기능 구현 후 CHANGELOG 항목만 추가한다.
- 문서와 버전 갱신은 모든 모듈을 `dev`에 통합한 뒤 한 번에 수행한다.
- 통합 테스트 후 `dev`에서 버전을 갱신하고 `main`에 병합한다.
- 버전 업데이트 시 README, CHANGELOG, `docs/whitepaper/index.md`, `pipeline/__init__.py`를 함께 수정한다.

- 시맨틱 버전 관리(SemVer)를 사용한다. 버전은 `MAJOR.MINOR.PATCH` 형식이다.
- 버전 변경은 `tools/bump_version.py` 스크립트로 수행한다.
- 최신 버전이 감지되면 requirements.txt와 `config/.pre-commit-config.yaml`을 갱신해
  패키지 모듈 버전을 최신 상태로 유지한다.
- requirements.txt에는 패키지 버전을 `>=` 형식으로 작성해 항상 최신 패키지를 설치한다.

- 코드 내 주석: 복잡한 로직이나 중요한 결정 사항에 대해서는 코드 내에 간결하고 명확한 주석을 남깁니다.
- API 문서화: 모듈 간 또는 외부와의 인터페이스가 되는 API는 명확하게 문서화합니다. (예: Swagger/OpenAPI 사용)

- 지속적 통합 (Continuous Integration, CI): Dev 브랜치에 코드가 푸시되거나 Pull Request가 생성될 때마다 자동으로 빌드 및 테스트를 실행하는 CI 파이프라인을 구축한다. (예: GitHub Actions, Jenkins) 이를 통해 통합 과정에서 발생하는 문제를 조기에 발견할 수 있다.

- 코드 리뷰: Dev 브랜치로 병합하기 전에 동료 개발자의 코드 리뷰를 필수로 진행한다. 이를 통해 잠재적인 버그를 찾고 코드 품질을 향상시킬 수 있다.

- README.md:
   * 프로젝트 루트: 프로젝트 전체에 대한 개요, 주요 기술 스택, 설치 및 실행 방법, 기여 방법 등을 작성합니다.
   * 각 모듈 폴더: 해당 모듈의 역할, 사용법, 주요 API, 의존성 등을 상세히 기술합니다.
- CHANGELOG.md: 프로젝트 또는 각 중요 모듈의 루트에 CHANGELOG.md 파일을 만들어 버전별 변경 사항(새로운 기능, 버그 수정, 주요 변경점 등)을 시간 순서대로 기록합니다. 이는 다른 개발자들이 변경 이력을 쉽게 파악하는 데 도움을 줍니다. (Keep a Changelog 형식 참고)
- `pipeline` 폴더는 `mapping`, `forecasting`, `datamart` 하위 패키지로 분리하고 테스트 역시 같은 구조를 따른다.

- 프로그램 명칭은 `MAPCAST`이며 mapping·forecasting 모듈을 포함한다.
- 메뉴 스크립트는 `tools/cli_menu.py` 공통 모듈을 이용해 중복을 최소화한다.
- 기능 개발 시 향후 유지보수와 확장성을 위해 각 기능을 모듈 단위로 설계한다.
- 공통적으로 쓰이는 로직은 별도의 공통 모듈에 구현해 모든 모듈에서 이를 활용한다.
## Codex 개발·문서화 핵심 지침 (One-Pager)
1. **코드 품질 · 구조**
   - 단일 책임 함수·클래스로 모듈화하고 계층을 명확히 분리합니다.
2. **명명 규칙**
   - 변수·함수·파일은 하나의 스타일만 사용하며 의미 전달형 이름을 권장합니다.
   - 코드 내 한글 식별자는 최소화하되 주석과 문서는 한글을 우선합니다.
3. **주석·문서**
   - 함수·클래스 docstring에는 한글 요약과 인자/반환 설명을 포함합니다.
   - 인라인 주석은 "무엇"보다 "왜" 필요한지 기록합니다.
   - README.md와 CHANGELOG.md를 항상 최신 상태로 유지합니다.
   - 아키텍처·데이터 모델 원본은 Markdown(+Mermaid/PlantUML)으로 저장합니다.
4. **로깅**
   - 로그 메시지는 기본적으로 한글을 사용합니다.
5. **배포 · 버전**
   - Semantic Versioning 태그를 사용하고 자동 릴리스 노트를 생성합니다.
   - 환경 변수나 시크릿은 Git에 포함하지 않습니다.
6. **백서(White Paper) 관리**
   - 백서 버전은 코드 태그(vX.Y.Z)와 동일하게 유지합니다.
   - Revision History는 CHANGELOG.md와 연동해 백서 앞에 요약을 둡니다.
   - 기능 PR 병합 시 docs/whitepaper/ 문서를 함께 수정합니다.
   - 태그 푸시 시 MkDocs로 whitepaper-vX.Y.Z.pdf를 생성해 첨부합니다.
7. 필요 시 AGENTS.md를 보완해 의도를 명확히 고정합니다.

## 시작 체크리스트 (신규 레포)
1. `git init` 후 `.gitignore` 작성
2. 의존성 매니저 초기화(poetry/npm init 등)
3. 기본 폴더 생성과 README 초안 작성
4. pre-commit(lint/format) 훅 설정
5. CI 템플릿 복사 후 첫 빌드 통과 확인
6. 본 지침 링크를 `CONTRIBUTING.md`에 포함

> **원칙**: 문서와 테스트가 없는 코드는 머지할 수 없습니다.
> **목표**: 유지보수 용이성, 온보딩 1일, 릴리스와 문서 완전 동기화
