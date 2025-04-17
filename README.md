# End-to-End ML Development with SageMaker Studio and MLflow

---

## 목차

- [실습 URL](#실습-url)
- [데이터 세트](#데이터-세트)
- [모듈 구성](#모듈-구성)
- [워크샵 환경 설정](#워크샵-환경-설정)
- [모듈 1: 모델 구축 및 학습](#모듈-1-모델-구축-및-학습)
- [모듈 2: 모델 배포](#모듈-2-모델-배포)
- [모듈 3: 완전한 배포 파이프라인 만들기](#모듈-3-완전한-배포-파이프라인-만들기)
- [모듈 4: API Gateway 및 Lambda로 HTTP API 구축](#모듈-4-api-gateway-및-lambda로-http-api-구축)
- [모듈 5: 웹 클라이언트에서 HTTP API 호출](#모듈-5-웹-클라이언트에서-http-api-호출)

---

## 실습 URL

- [AWS 공식 워크샵](https://catalog.workshops.aws/scale-complete-ml-development-with-amazon-sagemaker-studio/en-US)

---

## 데이터 세트

- [AI4I 2020 예측 유지 관리 데이터 세트](https://archive.ics.uci.edu/ml/datasets/AI4I+2020+Predictive+Maintenance+Dataset)
    - **특징**: 다변량, 시계열, 합성 데이터 (10,000개 레코드, 14개 피처)
    - **목표**: 기계 고장 여부(이진 분류) 예측
    - **모델**: 로지스틱 회귀 (XGBoost 활용)
    - **비고**: 실제 현업 적용 전 단계의 단순화된 예제 데이터

---

## 모듈 구성

| 모듈 번호 | 주요 내용 |
|-----------|-----------|
| **모듈 1** | SageMaker Studio JupyterLab에서 실험, feature engineering, XGBoost 모델 구축 및 학습 |
| **모듈 2** | Code Editor에서 모델을 추론 엔드포인트로 배포 |
| **모듈 3** | SageMaker Pipeline으로 데이터 다운로드, feature engineering, 학습, 등록, 배포까지의 end-to-end 파이프라인 구축 |
| **모듈 4** | API Gateway와 Lambda를 통한 HTTP API 구축 및 엔드포인트 호출 |
| **모듈 5** | 웹 클라이언트에서 HTTP API 호출 및 추론 수행 |

---

## 워크샵 환경 설정

- **IAM user/role**: Studio 및 Lambda 등 서비스 권한 관리
- **네트워킹 모드**: `direct_mode`, `vpc_mode` 지원
- **CloudFormation**: 인프라 자동 생성
- **SageMaker Studio 시작**
    - Region, Domain, User profile, Execution role 등 개념 정립

---

## 모듈 1: 모델 구축 및 학습

### 1. JupyterLab Space 생성 및 환경 설정

- 이름: `jupyterlab-space`
- 인스턴스: `ml.m5.large`
- 이미지: `SageMaker Distribution 1.11.0`
- 스토리지: 50GB

### 2. 코드 저장소 복제 및 노트북 실행

```bash
git clone https://github.com/aws-samples/amazon-sagemaker-build-train-deploy.git
```

- 설치 노트북: `00-start-here.ipynb`
- 빌드/학습 노트북: `01_build_and_train.ipynb`

### 3. 데이터 전처리 및 Feature Engineering

- **전처리 절차**
    - 열 선택, 범주형/수치형 구분
    - 타깃: 'Machine failure'
    - 데이터 분할: train(80%), validation(10%), test(10%)
    - `ColumnTransformer`로 one-hot 인코딩 및 표준화
    - 전처리 모델 저장: `/opt/ml/model/sklearn_model.joblib`

```python
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
```

### 4. 모델 학습

- XGBoost 활용, 주요 하이퍼파라미터: `eta=0.3`, `max_depth=8`
- 테스트 결과 예시:

```
===Metrics for Test Set===

Predictions  0.0  1.0   All
Actuals                    
0            961    5   966
1             10   24    34
All          971   29  1000

Accuracy Model A: 98.50%
Precision Model A: 0.83
Recall Model A: 0.71
AUC A: 0.94
```

### 5. 실험 관리

- MLflow experiment, S3 저장 위치, Training jobs 확인

---

## 모듈 2: 모델 배포

### 1. Code Editor Space 생성 및 환경 설정

- 이름: `code-editor-space`
- 인스턴스: `ml.m5.large`
- 이미지: `SageMaker Distribution 1.11.0`
- 스토리지: 50GB

### 2. 추론 엔드포인트 배포

- 주요 스크립트: `deploy.py`
- 모델 S3 위치 지정 후 배포 코드 실행

```python
deploy_model(pipeline_model, project_prefix, "ml.m5.xlarge", wait=False)
```

### 3. 엔드포인트 상태 및 테스트

- SageMaker Studio/콘솔에서 상태 확인
- `test.py`로 추론 테스트

```bash
python3 test.py ENDPOINT_NAME
```

---

## 모듈 3: 완전한 배포 파이프라인 만들기

### 1. SageMaker Pipeline 정의

- `@step` 데코레이터로 각 단계(데이터 처리, 학습, 평가, 등록, 배포) 구현
- 주요 파일: `pipeline.py`

### 2. 파이프라인 실행 및 모니터링

- 실행 방법: VSCode에서 Python 파일 실행 또는 노트북 활용
- 진행 상황: SageMaker Studio > Pipelines, S3, MLflow에서 확인

---

## 모듈 4: API Gateway 및 Lambda로 HTTP API 구축

### 1. Lambda 함수 생성 및 코드 배포

- 이름: `end-to-end-ml-lambda-function`
- 런타임: Python 3.12
- Execution Role: SageMaker 엔드포인트 호출 권한 필요
- 환경 변수: `SAGEMAKER_ENDPOINT_NAME`
- Lambda 코드 예시:

```python
def lambda_handler(event, context):
    # ... 생략 ...
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME, ContentType='text/plain', Body=turbine_data)
    # ... 생략 ...
```

### 2. API Gateway HTTP API 생성 및 람다 트리거 연결

- 엔드포인트 예시:  
  `https://1234567890.execute-api.us-east-1.amazonaws.com/default/end-to-end-ml-lambda-function`

---

## 모듈 5: 웹 클라이언트에서 HTTP API 호출

```python
import requests

api_url = "https://h1hlcxkexd.execute-api.us-east-1.amazonaws.com/default/end-to-end-ml-lambda-function"
headers = {"Content-Type": "text/plain"}
payload = "L,298.4,308.2,1582,70.7,216"

response = requests.post(api_url, headers=headers, data=payload)
if response.status_code == 200:
    print("성공:", response.text)
else:
    print(f"실패: {response.status_code}, 응답: {response.text}")
```

---

