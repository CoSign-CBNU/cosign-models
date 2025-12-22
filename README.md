# Cosign Models - 수어 인식 시스템

한국 수어를 실시간으로 인식하는 머신러닝 기반 시스템입니다. MediaPipe를 활용하여 손과 신체 키포인트를 추출하고, 다양한 머신러닝 모델로 수어를 분류합니다.

## 목차

- [프로젝트 개요](#프로젝트-개요)
- [시스템 구성](#시스템-구성)
- [설치 방법](#설치-방법)
- [프로젝트 작동 순서](#프로젝트-작동-순서)
- [한글 라벨 설정](#한글-라벨-설정)
- [모델 학습](#모델-학습)
- [실시간 인식](#실시간-인식)
- [웹 서버 실행](#웹-서버-실행)
- [프로젝트 구조](#프로젝트-구조)
- [트러블슈팅](#트러블슈팅)

## 프로젝트 개요

### 주요 기능

- **키포인트 추출**: MediaPipe를 사용한 손/신체 랜드마크 추출
- **Idle/Active 감지**: 실제 수어 동작 구간 자동 탐지
- **다양한 모델 지원**: Random Forest, Linear SVC, MLP, Logistic Regression
- **실시간 인식**: 웹캠을 통한 실시간 수어 인식
- **웹 서버**: FastAPI + WebSocket 기반 웹 연동
- **한글 라벨**: 숫자 라벨을 의미 있는 한글 단어로 표시

### 지원 모델

현재 15개 한국 수어 단어 인식 가능:

- 운전면허, 골키퍼, 구경, 성토, 권투, 상처, 병, 환자, 간, 위, 가루약, 감염병, 중병, 물약, 팔

## 시스템 구성

```
수어 영상 → 키포인트 추출 → 데이터셋 생성 → 모델 학습 → 실시간 인식/웹 서버
```

## 설치 방법

### 필수 요구사항

- Python 3.8 이상
- 웹캠 (실시간 인식 시)

### 패키지 설치

```bash
pip install -r requirements.txt
```

또는 수동 설치:

```bash
pip install opencv-python mediapipe numpy tqdm pillow scikit-learn joblib fastapi uvicorn websockets python-multipart
```

## 프로젝트 작동 순서

### 1단계: 키포인트 추출

수어 영상에서 MediaPipe를 사용하여 손과 신체 키포인트를 추출합니다.

```bash
python extract_keypoints.py
```

**실행 결과:**

- `dataset_out/` 폴더에 각 라벨(1, 2, 3, ...)별로 `.npy`, `.csv` 파일 생성
- 각 영상당 60프레임의 키포인트 시퀀스 추출
- Idle 구간을 자동으로 제외하고 Active 구간만 추출

<img width="319" height="257" alt="image" src="https://github.com/user-attachments/assets/e9399581-8293-4fb0-9d4b-42167cdfdb03" />

**데이터 폴더 구조:**

```
data/
  1/  # 라벨 1 (운전면허)
    video1.mp4
    video2.mp4
  2/  # 라벨 2 (골키퍼)
    video1.mp4
    video2.mp4
  ...
```

### 2단계: 데이터셋 생성

추출된 키포인트를 학습 가능한 형태로 변환합니다.

```bash
python build_dataset.py
```

**실행 결과:**

- `dataset_out/X_seq.npy`: 전체 시퀀스 데이터 (N, 60, feat_dim)
- `dataset_out/y.npy`: 라벨 데이터 (N,)
- `dataset_out/label_map.json`: 라벨 매핑 정보 (한글 포함)

### 3단계: 모델 학습

여러 모델 중 하나를 선택하여 학습합니다.

#### Random Forest (추천)

```bash
python train_rf.py
# 또는 전체 데이터셋 사용
python train_rf_all.py
```

#### Linear SVC (Calibrated)

```bash
python train_linearsvc_calibrated.py
```

#### 기타 모델

```bash
python train_mlp.py      # MLP
python train_logistic_all.py  # Logistic Regression
```

**실행 결과:**

- `dataset_out/*.pkl`: 학습된 모델 파일
- 5-fold 교차검증 성능 출력
- 학습 완료 후 모델 저장

### 4단계: 실시간 인식 또는 서버 실행

#### 로컬 실시간 인식

```bash
python realtime_sign.py
```

- 웹캠으로 수어 동작 수행
- 인식된 한글 단어가 화면에 표시
- ESC 키로 종료

#### 웹 서버 실행

```bash
python server.py
```

- FastAPI 서버 시작 (포트: 8000)
- WebSocket 엔드포인트: `ws://localhost:8000/ws/recognize`
- 프론트엔드와 연동하여 사용 (자세한 내용은 [README_WEB_INTEGRATION.md](README_WEB_INTEGRATION.md) 참조)

## 한글 라벨 설정

### 라벨 매핑 파일 수정

`word_labels.json` 파일을 수정하여 숫자 라벨을 한글 단어로 매핑할 수 있습니다:

```json
{
  "1": "운전면허",
  "2": "골키퍼",
  "3": "구경",
  ...
}
```

### 라벨 변경 후 작업

라벨을 수정한 후에는 다음 단계를 다시 실행해야 합니다:

```bash
# 1. 데이터셋 재생성
python build_dataset.py

# 2. 모델 재학습
python train_rf_all.py

# 3. 실시간 인식 테스트
python realtime_sign.py
```

자세한 내용은 [한글*라벨*사용법.md](한글_라벨_사용법.md)를 참조하세요.

## 모델 학습

### 학습 파라미터

모델 학습 시 조정 가능한 주요 파라미터:

- `SEQ_LEN`: 시퀀스 길이 (기본값: 60)
- `MAX_HANDS`: 처리할 손 개수 (1 또는 2)
- `USE_BODY`: 상체 키포인트 사용 여부
- `USE_ELBOWS`: 팔꿈치 포함 여부
- `USE_HIPS`: 엉덩이 포함 여부
- `USE_LIMB_ANGLES`: 팔 벡터/각도 피처 사용 여부

### 모델 비교

여러 모델의 성능을 비교하려면:

```bash
python compare_models.py
```

### 테스트 데이터 평가

별도의 테스트 데이터셋으로 모델 평가:

```bash
# 테스트 데이터에서 키포인트 추출
python extract_test_keypoints.py

# 테스트 데이터셋 생성
python build_test_repr.py

# 모델 평가
python eval_rf_on_new.py
```

## 실시간 인식

### 주요 파라미터

실시간 인식 시 조정 가능한 파라미터 (`realtime_sign.py`):

#### Idle/Active 감지

- `IDLE_VEL_THR`: 손목 속도 임계값 (0.018)
- `MIN_ACTIVE_FRAMES`: 최소 active 프레임 수 (10)
- `IDLE_END_N`: 연속 idle 프레임 수 (5)

#### 오인식 방지

- `MIN_CONFIDENCE`: 최소 신뢰도 (0.15)
- `COOLDOWN_TIME`: 인식 후 대기 시간 (1.0초)
- `STABLE_IDLE_COUNT`: 다음 인식 가능한 idle 프레임 수 (5)

### 사용 방법

1. `python realtime_sign.py` 실행
2. 웹캠 앞에서 수어 동작 수행
3. 화면에 인식 결과 표시
4. ESC 키로 종료

### 디버그 모드

Idle 상태를 시각적으로 확인하려면:

```bash
python debug_idle.py
```

## 웹 서버 실행

### 서버 시작

```bash
python server.py
```

서버가 시작되면:

- 주소: `http://localhost:8000`
- WebSocket: `ws://localhost:8000/ws/recognize`

### API 엔드포인트

#### WebSocket: `/ws/recognize`

**프레임 전송 (클라이언트 → 서버):**

```json
{
  "frame": "base64_encoded_jpeg_image"
}
```

**인식 결과 (서버 → 클라이언트):**

```json
{
  "word": "안녕",
  "confidence": 0.85,
  "timestamp": 1234567890.123
}
```

#### Health Check: `GET /`

서버 상태 확인

### 프론트엔드 연동

웹 프론트엔드와 연동하는 방법은 [README_WEB_INTEGRATION.md](README_WEB_INTEGRATION.md)를 참조하세요.

## 프로젝트 구조

```
cosign-models/
│
├── data/                          # 학습 영상 데이터
│   ├── 1/                         # 라벨 1 영상들
│   ├── 2/                         # 라벨 2 영상들
│   └── ...
│
├── data_test/                     # 테스트 영상 데이터
│   └── ...
│
├── dataset_out/                   # 키포인트 & 모델 출력
│   ├── 1/                         # 라벨별 .npy, .csv
│   ├── X_seq.npy                  # 전체 시퀀스
│   ├── y.npy                      # 라벨
│   ├── label_map.json             # 라벨 매핑
│   └── *.pkl                      # 학습된 모델들
│
├── extract_keypoints.py           # 키포인트 추출
├── extract_test_keypoints.py     # 테스트용 키포인트 추출
├── build_dataset.py               # 데이터셋 생성
├── build_test_repr.py             # 테스트 데이터셋 생성
│
├── train_rf.py                    # Random Forest 학습
├── train_rf_all.py                # RF 전체 데이터 학습
├── train_linearsvc.py             # Linear SVC 학습
├── train_linearsvc_calibrated.py # Linear SVC (Calibrated)
├── train_mlp.py                   # MLP 학습
├── train_logistic_all.py          # Logistic Regression
│
├── realtime_sign.py               # 실시간 수어 인식
├── realtime_sign2.py              # 실시간 수어 인식 (변형)
├── debug_idle.py                  # Idle 상태 디버깅
│
├── server.py                      # FastAPI 웹 서버
├── compare_models.py              # 모델 성능 비교
├── eval_rf_on_new.py              # 테스트 데이터 평가
│
├── word_labels.json               # 한글 라벨 매핑
├── requirements.txt               # 의존성 패키지
├── README.md                      # 본 파일
├── README_WEB_INTEGRATION.md      # 웹 연동 가이드
└── 한글_라벨_사용법.md             # 한글 라벨 가이드
```

## 트러블슈팅

### 웹캠이 열리지 않음

```python
# realtime_sign.py에서 CAM_INDEX 변경
CAM_INDEX = 1  # 또는 다른 번호
```

### 모델 파일이 없음

모델을 학습했는지 확인:

```bash
python train_rf_all.py
```

### 키포인트 추출 실패

- MediaPipe가 손/신체를 감지하지 못하는 경우
- 조명이 충분한지 확인
- 카메라와의 거리가 적절한지 확인

### 한글이 깨짐

Windows에서 실행 시 콘솔 인코딩 문제:

```bash
# 명령 프롬프트에서
chcp 65001
python realtime_sign.py
```

### 인식 정확도가 낮음

1. 더 많은 학습 데이터 추가
2. 다양한 각도/환경에서 촬영
3. 파라미터 조정:
   - `MIN_CONFIDENCE` 높이기
   - `IDLE_VEL_THR` 조정
4. 다른 모델 시도

### 오인식이 많이 발생

`realtime_sign.py`에서 파라미터 조정:

```python
MIN_CONFIDENCE = 0.2      # 높이기
COOLDOWN_TIME = 1.5       # 늘리기
STABLE_IDLE_COUNT = 8     # 늘리기
```

## 참고 자료

- **MediaPipe Hands**: https://google.github.io/mediapipe/solutions/hands
- **MediaPipe Pose**: https://google.github.io/mediapipe/solutions/pose
- **Scikit-learn**: https://scikit-learn.org/
- **FastAPI**: https://fastapi.tiangolo.com/

## 라이선스

이 프로젝트는 교육 및 연구 목적으로 제공됩니다.

## 문의

프로젝트 관련 문의사항이나 버그 리포트는 GitHub Issues를 통해 제출해 주세요.
