# OpenSafe AI — 서울 창업 리스크 진단 MVP

서울시 상권분석서비스의 행정동 단위 점포·추정매출·길단위 유동인구 CSV를 결합한 로컬 대시보드입니다. 희망 업종과 행정동을 고르면 현재 상권 리스크 지수, 다음 분기 폐업률 AI 예측, 매출 추이, 주요 위험 신호, 대안 상권을 확인할 수 있습니다.

## 실행

이 폴더에서 아래 명령을 실행한 뒤, 브라우저에서 `http://127.0.0.1:8765`를 여세요.

```powershell
python app.py --open
```

기본 Python이 없다면 다음처럼 Codex에 포함된 Python을 사용해도 됩니다.

```powershell
& 'C:\Users\smho0\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' app.py --open
```

서버를 종료하려면 터미널에서 `Ctrl+C`를 누르세요.

## 데이터·지수 기준

- 분석 기준 분기: `2026년 1분기` (`20261`)
- 비교 단위: **같은 서비스 업종 안에서 서울 행정동 간 상대 비교**
- 학습 기반 리스크 지수(0~100): 과거 분기로 학습한 다음 분기 폐업률 예측값을 같은 업종의 서울 행정동 안에서 순위화한 점수
- 점포당 추정매출: 행정동·업종의 추정매출 총액 ÷ 전체 점포 수

## AI 예측 모델

- 모델: NumPy로 구현한 **Ridge 회귀 머신러닝 모델**
- 예측 목표: 현재 분기의 관측치를 사용한 **다음 분기 행정동·업종 폐업률**
- 입력: 최근 폐업·개업률, 점포 수·점포 밀도, 점포당 매출·거래건수, 매출 증감, 프랜차이즈 비중, 업종 기준선
- 검증: 미래 분기를 학습에 섞지 않는 시간 순서 홀드아웃(`2025년 4분기 → 2026년 1분기`)으로 검증
- 대시보드의 예측 범위: 홀드아웃 평균절대오차(MAE)를 예측값에 더하고 뺀 참고 범위
- 리스크 지수 검증: 홀드아웃 분기에서 동일 업종 상위 25% 위험군과 하위 25% 위험군의 실제 다음 분기 폐업률을 비교

## 안정성 × 시장 진입 4분면

- 세로축(안정성): AI가 예측한 다음 분기 폐업률의 동일 업종 내 상대 순위
- 가로축(시장 진입): 개업률 − 폐업률인 **순점포 증감률**의 동일 업종 내 상대 위치
- 영역: 안정적 성장, 경쟁 치열, 성숙·안정, 수축 위험 / 틈새 검토

개업률이 높아도 폐업률이 더 높을 수 있으므로, 시장 기회는 개업률만으로 판단하지 않습니다. 수축 위험 / 틈새 검토 영역은 점포당 매출이 업종 중앙값 이상이고 최근 매출 추세가 유지될 때만 틈새 검토로 표시하며, 그렇지 않으면 수축 위험으로 표시합니다.

이 지수와 AI 예측은 공개 상권 통계에 기반한 사전 검토 도구입니다. 특정 가게의 폐업 확률이나 실제 폐업 원인을 예측·판정하지 않습니다. 임대료, 보증금, 점포 면적, 점주 경력, 상권별 접근성 등 미포함 요인은 계약 전 별도로 확인해야 합니다.

## 데이터 재생성

`data/dashboard-data.json`은 제공된 세 CSV에서 생성되었습니다. 새 분기 데이터로 갱신하려면 다음 명령을 사용하세요. 세 CSV는 한 폴더 안에 넣고, 파일은 CP949 인코딩·원본 열 구조를 유지해야 합니다.

```powershell
python build_dashboard.py --input-dir 'C:\Users\smho0\Downloads' --output 'data\dashboard-data.json'
```

현재 대시보드는 외부 API나 별도 패키지 없이 동작합니다.

## Pandas 기반 데이터 분석

발표용 전처리·분석 근거는 `pandas_analysis.py`로 재현할 수 있습니다. 이 단계는 배포용 대시보드와 분리되어 있으므로 Vercel에 Pandas를 설치할 필요가 없습니다.

```powershell
# Python에 Pandas가 없다면 한 번만 설치
python -m pip install pandas

python pandas_analysis.py --input-dir 'C:\Users\smho0\Downloads' --dashboard-data 'data\dashboard-data.json' --output-dir 'analysis'
```

실행하면 `analysis` 폴더에 다음 파일이 생성됩니다.

- `data_quality_summary.csv`: 원본·결합 데이터의 행 수, 결측치, 키 중복 점검
- `quarterly_market_summary.csv`: 분기별 개·폐업률, 순점포 증감률, 점포당 매출 요약
- `quadrant_summary.csv`: 4분면별 점포 수와 평균 개·폐업률·매출·AI 예측 결과
- `analysis_report.md`: 발표에 바로 활용할 수 있는 분석 과정과 핵심 수치

## 12개월 운영 시나리오

대시보드 하단의 **12개월 운영 시나리오**에서 다음 가정을 입력할 수 있습니다.

- 월 임대료·기타 월 고정비
- 변동비율
- 예상 객단가·월 영업일
- 상권 평균 대비 매출 가정 조정치
- 초기 운영예산

선택 행정동·업종의 점포당 추정매출을 출발점으로 월매출, 손익분기 매출, 일평균 필요 거래 수, 적자 시 예산 여력, 12개월 운영 여력 지수를 계산합니다. 이는 사용자 가정과 행정동·업종 평균을 결합한 손익 시나리오이며 개별 점포의 실제 매출이나 생존을 보장하지 않습니다.

## 선택적 GenAI 창업 코치

코치는 다음 우선순위로 동작합니다.

1. 실행 중인 **Ollama 로컬 모델** — API 키 없이 GenAI 답변 생성
2. `GEMINI_API_KEY`가 설정된 Gemini API
3. 외부 모델 없이 동작하는 **규칙 기반 코치**

기본값은 `GENAI_PROVIDER=auto`입니다. 로컬 모델만 강제하려면 `GENAI_PROVIDER=ollama`, 규칙 기반만 쓰려면 `GENAI_PROVIDER=rules`를 설정할 수 있습니다.

### 키 없는 로컬 GenAI: Ollama

Ollama를 설치하고 한국어 응답이 가능한 모델을 하나 내려받아 실행하세요. 설치된 모델 이름은 `ollama list`에서 확인할 수 있습니다.

```powershell
# 예: 이미 설치한 모델 이름을 그대로 설정합니다.
$env:OLLAMA_MODEL = 'YOUR_INSTALLED_MODEL_NAME'
$env:GENAI_PROVIDER = 'ollama' # 선택 사항, auto가 기본값입니다.
python app.py --open
```

Ollama가 기본 로컬 주소(`http://127.0.0.1:11434`)에서 실행 중이면 앱이 자동 감지합니다. 앱은 안전을 위해 `127.0.0.1`·`localhost`의 Ollama만 연결하며, 선택 상권의 집계 지표와 시나리오 값만 로컬 모델에 전달합니다. 모델이 없거나 Ollama가 실행되지 않으면 규칙 기반 코치로 자동 전환됩니다.

### Gemini API 사용

Gemini API를 연결하려면 실행 전 PowerShell에서 환경변수를 설정하세요. API 키는 코드나 브라우저에 넣지 않습니다.

```powershell
$env:GEMINI_API_KEY = 'YOUR_API_KEY'
# 선택 사항: 기본값은 gemini-3.5-flash입니다.
$env:GEMINI_MODEL = 'gemini-3.5-flash'
python app.py --open
```

키가 있으면 서버가 Gemini API에 구조화된 JSON 응답을 요청해 GenAI 코치 결과를 만듭니다. 전달되는 내용은 선택 상권의 집계 지표와 사용자가 입력한 시나리오 값뿐이며, 프롬프트는 제공된 수치 밖의 사실·법률·개별 점포 성공 보장을 만들지 않도록 제한합니다. API가 실패하면 규칙 기반 코치로 되돌아갑니다.

## GitHub·Vercel 배포

이 폴더 자체를 Git 저장소 루트로 사용하도록 구성했습니다. 원본 CSV는 저장소에 넣지 않으며, 배포에 필요한 `data/dashboard-data.json`만 추적합니다.

### 1. GitHub 저장소 생성

GitHub에서 빈 저장소를 만든 뒤, 이 폴더에서 아래 명령을 실행합니다.

```powershell
git init
git add .
git commit -m "Initial OpenSafe AI dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_GITHUB_ID/open-safe-ai.git
git push -u origin main
```

`.gitignore`은 API 키가 들어갈 수 있는 `.env` 파일, Vercel 연결 정보, Python 캐시를 제외합니다. `GEMINI_API_KEY`는 절대로 GitHub에 커밋하지 마세요.

### 2. Vercel 연결

1. Vercel Dashboard에서 **New Project**를 누릅니다.
2. 방금 만든 GitHub 저장소를 Import합니다.
3. 이 저장소가 프로젝트 루트이므로 Root Directory는 `.`으로 둡니다.
4. Framework Preset은 **Other**를 선택하고 Deploy합니다.

`index.html`과 데이터 JSON은 정적으로 배포되고, `api/scenario.py`·`api/coach.py`는 Vercel Python Functions로 배포됩니다. `vercel.json`은 두 함수가 `data/dashboard-data.json`을 읽도록 포함합니다.

### 3. Vercel 환경변수

Vercel Project Settings → Environment Variables에 필요한 값만 설정합니다.

```text
# 외부 GenAI 없이 배포할 때
GENAI_PROVIDER=rules

# Gemini GenAI 코치를 쓸 때
GENAI_PROVIDER=gemini
GEMINI_API_KEY=YOUR_API_KEY
GEMINI_MODEL=gemini-3.5-flash
```

Vercel에서는 사용자 PC의 Ollama에 접근할 수 없으므로 Ollama 기반 코치는 동작하지 않습니다. 배포판에서는 Gemini GenAI 코치 또는 규칙 기반 코치를 사용합니다.

### 4. 수정 확인 흐름

```powershell
git checkout -b feature/my-change
# 파일 수정
git add .
git commit -m "Describe the change"
git push -u origin feature/my-change
```

Vercel은 브랜치 push마다 Preview URL을 만들고, 확인 후 `main`에 병합하면 운영 URL을 업데이트합니다.
