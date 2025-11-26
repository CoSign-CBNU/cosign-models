# 수어 인식 시스템 - 웹 연동 가이드

## 시스템 구성

1. **백엔드 서버** (test2_60f_extract/server.py)

   - FastAPI + WebSocket
   - Mediapipe로 수어 인식
   - 학습된 RF 모델 사용

2. **프론트엔드** (front/)
   - React + Vite
   - 웹캠으로 영상 캡처
   - WebSocket으로 실시간 통신

## 설치 및 실행 방법

### 1. 백엔드 서버 실행

```bash
# test2_60f_extract 디렉토리로 이동
cd test2_60f_extract

# 가상환경 생성 (선택사항이지만 권장)
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Mac/Linux

# 필요한 패키지 설치
pip install -r requirements.txt

# 서버 실행
python server.py
```

서버가 정상적으로 시작되면:

- http://localhost:8000 에서 실행됩니다
- WebSocket 엔드포인트: ws://localhost:8000/ws/recognize

### 2. 프론트엔드 실행

```bash
# front 디렉토리로 이동
cd front

# 패키지 설치 (최초 1회만)
npm install

# 개발 서버 실행
npm run dev
```

브라우저에서 http://localhost:3000 접속

## 사용 방법

1. 프론트엔드 웹페이지에서 "인식 시작" 버튼 클릭
2. 카메라 권한 허용
3. 수어 동작 수행
4. 오른쪽에 인식 결과가 표시됨

## 작동 원리

1. **프론트엔드** (Camera 컴포넌트)

   - 웹캠에서 30fps로 프레임 캡처
   - 각 프레임을 JPEG로 압축 후 Base64 인코딩
   - WebSocket으로 서버에 전송

2. **백엔드** (server.py)

   - WebSocket으로 프레임 수신
   - Mediapipe로 손/신체 랜드마크 추출
   - idle/active 상태 감지
   - active 상태에서 프레임 수집
   - idle로 전환 시 수집된 시퀀스로 수어 분류
   - 결과를 WebSocket으로 프론트엔드에 전송

3. **프론트엔드** (Main 컴포넌트)
   - WebSocket으로 결과 수신
   - TranslationResult에 표시

## 주요 파일

### 백엔드

- `server.py`: FastAPI WebSocket 서버
- `realtime_sign2.py`: 원본 로컬 인식 스크립트 (참고용)
- `dataset_out/rf_model_all.pkl`: 학습된 모델
- `dataset_out/label_map.json`: 라벨 매핑

### 프론트엔드

- `src/services/websocket.js`: WebSocket 클라이언트
- `src/components/camera/Camera.jsx`: 웹캠 + 프레임 전송
- `src/pages/Main/Main.jsx`: 메인 페이지 (WebSocket 관리)

## 트러블슈팅

### 백엔드 서버 연결 실패

- 백엔드 서버가 실행 중인지 확인
- 방화벽에서 8000 포트 허용
- 프론트엔드 websocket.js의 서버 URL 확인

### 카메라가 작동하지 않음

- 브라우저에서 카메라 권한 허용
- HTTPS 환경인지 확인 (localhost는 HTTP 가능)
- 다른 프로그램에서 카메라를 사용 중인지 확인

### 인식이 안 됨

- 백엔드 콘솔에서 vmag (손목 속도) 값 확인
- 충분히 큰 동작으로 수어 수행
- 조명이 충분한지 확인
- 손과 상체가 프레임 안에 들어오는지 확인

### 프레임 전송 지연

- 네트워크 대역폭 확인
- 프레임 압축 품질 조정 (websocket.js의 canvas.toBlob 두 번째 인자)
- 전송 주기 조정 (Camera.jsx의 setInterval 간격)

## 성능 최적화 팁

1. **프레임 전송 최적화**

   - JPEG 품질 낮추기 (현재 0.8)
   - 해상도 줄이기
   - 전송 주기 늘리기 (33ms → 66ms)

2. **백엔드 처리 최적화**

   - Mediapipe 감지 임계값 조정
   - 병렬 처리 추가 (asyncio)

3. **네트워크 최적화**
   - 바이너리 프로토콜 사용 (Base64 대신)
   - 프레임 차이 전송 (변화 있는 경우만)

## 환경 변수 설정 (선택)

프론트엔드에서 서버 URL을 환경 변수로 관리:

```javascript
// .env
VITE_WEBSOCKET_URL=ws://localhost:8000/ws/recognize

// Main.jsx
const WS_URL = import.meta.env.VITE_WEBSOCKET_URL || 'ws://localhost:8000/ws/recognize';
```

## 다음 단계

- [ ] 인식 정확도 향상 (더 많은 데이터로 학습)
- [ ] 다중 수어 연속 인식
- [ ] 문장 구성 기능
- [ ] 인식 이력 저장
- [ ] 사용자 피드백 수집
