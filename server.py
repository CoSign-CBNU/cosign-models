# server.py - FastAPI 백엔드 서버
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import mediapipe as mp
import joblib
import json
import base64
from typing import Optional

# ===== 설정 (realtime_sign.py와 동일) =====
SEQ_LEN = 60
MODEL_PATH = "./dataset_out/linearsvc_calibrated.pkl"
LABEL_MAP_PATH = "./dataset_out/label_map.json"

IDLE_VEL_THR = 0.018
MIN_ACTIVE_FRAMES = 10
IDLE_END_N = 5

# 오인식 방지 파라미터
MIN_CONFIDENCE = 0.15       # 최소 신뢰도
COOLDOWN_TIME = 0.5        # 인식 후 대기 시간 (초)
STABLE_IDLE_COUNT = 5     # 다음 인식 가능하려면 필요한 idle 프레임 수

MAX_HANDS = 2
USE_BODY = True
USE_ELBOWS = True
USE_HIPS = True
USE_LIMB_ANGLES = True
USE_VELOCITY = False

# ===== Pose 인덱스 =====
POSE_NOSE = 0
POSE_LSHO, POSE_RSHO = 11, 12
POSE_LELB, POSE_RELB = 13, 14
POSE_LWR, POSE_RWR = 15, 16
POSE_LHIP, POSE_RHIP = 23, 24

# ===== FastAPI 앱 =====
app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 프로덕션에서는 특정 origin으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ===== 전역 변수 =====
clf = None
id2label = None
mp_hands_instance = None
mp_pose_instance = None


# ===== 유틸 함수들 (realtime_sign2.py에서 복사) =====
def lm21_to_rel(hlm):
    """손 랜드마크 21개를 (21,3) wrist 상대좌표로"""
    k = np.zeros((21, 3), np.float32)
    for i, lm in enumerate(hlm.landmark):
        k[i] = (lm.x, lm.y, lm.z)
    wrist = k[0].copy()
    k[:, :2] -= wrist[:2]
    return k


def pose_to_arr(plm):
    """Pose 결과 → (33,3) 배열"""
    k = np.zeros((33, 3), np.float32)
    if plm:
        for i, lm in enumerate(plm.landmark):
            if i < 33:
                k[i] = (lm.x, lm.y, lm.z)
    return k


def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6:
        return v * 0
    return v / n


def _angle_cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6:
        return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))


def _count_body_targets():
    count = 4
    if USE_ELBOWS:
        count += 2
    if USE_HIPS:
        count += 3
    return count


def compute_feat_dim():
    hand_dim = (21 * 3) * (2 if MAX_HANDS == 2 else 1)
    body_dim = 0
    if USE_BODY:
        body_dim = _count_body_targets() * 3 * (2 if MAX_HANDS == 2 else 1)
    limb_dim = 12 if (USE_BODY and USE_LIMB_ANGLES) else 0
    vel_dim = 4 if USE_VELOCITY else 0
    return hand_dim + body_dim + limb_dim + vel_dim


def _body_rel_features_full(pose33, left_wrist_abs, right_wrist_abs):
    """상체 기준 상대피처"""
    nose = pose33[POSE_NOSE]
    lsho, rsho = pose33[POSE_LSHO], pose33[POSE_RSHO]
    shoulder_mid = (lsho + rsho) / 2.0
    targets = [nose, shoulder_mid, lsho, rsho]

    if USE_ELBOWS:
        lelb, relb = pose33[POSE_LELB], pose33[POSE_RELB]
        targets += [lelb, relb]

    hip_mid = None
    if USE_HIPS:
        lhip, rhip = pose33[POSE_LHIP], pose33[POSE_RHIP]
        hip_mid = (lhip + rhip) / 2.0
        targets += [lhip, rhip, hip_mid]

    shw = np.linalg.norm(lsho[:2] - rsho[:2])
    torso = 0.0
    if USE_HIPS and hip_mid is not None:
        torso = np.linalg.norm(shoulder_mid[:2] - hip_mid[:2])
    norm_scale = max((shw + torso) / 2.0, 1e-6)

    def rel(wrist_abs):
        if wrist_abs is None:
            return np.zeros(len(targets) * 3, np.float32)
        vecs = [(wrist_abs - t) / norm_scale for t in targets]
        return np.concatenate(vecs, axis=0).astype(np.float32)

    if MAX_HANDS == 2:
        l = rel(left_wrist_abs)
        r = rel(right_wrist_abs)
        return np.concatenate([l, r], axis=0), norm_scale
    else:
        tgt = left_wrist_abs if left_wrist_abs is not None else right_wrist_abs
        return rel(tgt), norm_scale


def _limb_features(pose33, lwr, rwr, norm_scale):
    """팔 벡터/각도 피처"""
    lsho, rsho = pose33[POSE_LSHO][:2], pose33[POSE_RSHO][:2]
    lelb, relb = pose33[POSE_LELB][:2], pose33[POSE_RELB][:2]

    if USE_HIPS:
        lhip, rhip = pose33[POSE_LHIP][:2], pose33[POSE_RHIP][:2]
        shoulder_mid = (lsho + rsho) / 2.0
        hip_mid = (lhip + rhip) / 2.0
        trunk = hip_mid - shoulder_mid
        if np.linalg.norm(trunk) < 1e-6:
            trunk = np.array([0.0, 1.0], np.float32)
    else:
        trunk = np.array([0.0, 1.0], np.float32)

    feats = []

    def arm_side(sho, elb, wr_abs):
        upper = (elb - sho) / norm_scale
        if wr_abs is None:
            lower = np.zeros(2, np.float32)
        else:
            lower = (wr_abs[:2] - elb) / norm_scale
        elbow_cos = _angle_cos(_unit(upper), _unit(lower))
        shoulder_cos = _angle_cos(_unit(trunk), _unit(upper))
        return np.concatenate(
            [upper, lower, np.array([elbow_cos, shoulder_cos], np.float32)]
        )

    feats.append(arm_side(lsho, lelb, lwr))
    feats.append(arm_side(rsho, relb, rwr))

    return np.concatenate(feats).astype(np.float32)


def extract_feature_and_speed(hands_results, pose_results, prev_lw, prev_rw, feat_dim):
    """1프레임에서 피처 + 손목 속도 계산"""
    if USE_BODY and pose_results and pose_results.pose_landmarks:
        pose33 = pose_to_arr(pose_results.pose_landmarks)
    else:
        pose33 = np.zeros((33, 3), np.float32)

    left_rel = np.zeros((21, 3), np.float32)
    right_rel = np.zeros((21, 3), np.float32)
    lw_abs = None
    rw_abs = None

    if (
        hands_results
        and hands_results.multi_hand_landmarks
        and hands_results.multi_handedness
    ):
        for hlm, handed in zip(
            hands_results.multi_hand_landmarks, hands_results.multi_handedness
        ):
            lr = handed.classification[0].label
            rel = lm21_to_rel(hlm)
            wrist_abs = np.array(
                [hlm.landmark[0].x, hlm.landmark[0].y, hlm.landmark[0].z], np.float32
            )
            if lr == "Left":
                left_rel = rel
                lw_abs = wrist_abs
            else:
                right_rel = rel
                rw_abs = wrist_abs

    if MAX_HANDS == 2:
        hand_feat = np.concatenate(
            [left_rel.reshape(-1), right_rel.reshape(-1)], axis=0
        )
    else:
        if lw_abs is not None:
            hand_feat = left_rel.reshape(-1)
        else:
            hand_feat = right_rel.reshape(-1)

    feat_parts = [hand_feat]

    norm_scale = 1.0
    if USE_BODY:
        body_rel, norm_scale = _body_rel_features_full(pose33, lw_abs, rw_abs)
        feat_parts.append(body_rel)

    if USE_BODY and USE_LIMB_ANGLES:
        limb = _limb_features(pose33, lw_abs, rw_abs, norm_scale)
        feat_parts.append(limb)

    feat = np.concatenate(feat_parts, axis=0).astype(np.float32)

    if feat.shape[0] != feat_dim:
        if feat.shape[0] < feat_dim:
            pad = np.zeros(feat_dim - feat.shape[0], np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        else:
            feat = feat[:feat_dim]

    lv = np.zeros(2, np.float32)
    rv = np.zeros(2, np.float32)
    if lw_abs is not None and prev_lw is not None:
        lv = (lw_abs[:2] - prev_lw[:2]) / max(norm_scale, 1e-6)
    if rw_abs is not None and prev_rw is not None:
        rv = (rw_abs[:2] - prev_rw[:2]) / max(norm_scale, 1e-6)

    vmag = max(np.linalg.norm(lv), np.linalg.norm(rv))

    return feat, float(vmag), lw_abs, rw_abs


# ===== 서버 시작 시 모델 로드 =====
@app.on_event("startup")
async def startup_event():
    global clf, id2label, mp_hands_instance, mp_pose_instance

    # 모델 로드
    clf = joblib.load(MODEL_PATH)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)
    # id2label에 한글 라벨이 저장되어 있음
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    # Mediapipe 인스턴스
    mp_hands_instance = mp.solutions.hands.Hands(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    mp_pose_instance = mp.solutions.pose.Pose(
        min_detection_confidence=0.5, min_tracking_confidence=0.5
    )

    print("✅ 모델, Mediapipe 인스턴스 로드 완료")
    print(f"📝 라벨: {id2label}")


# ===== 헬스체크 엔드포인트 =====
@app.get("/")
async def root():
    return {"status": "ok", "message": "Sign Language Recognition Server"}


# ===== WebSocket 엔드포인트 =====
@app.websocket("/ws/recognize")
async def websocket_recognize(websocket: WebSocket):
    await websocket.accept()
    print("🔌 WebSocket 연결됨")

    # 상태 변수
    state = "idle"  # "idle", "active", "cooldown"
    active_feats = []
    idle_count = 0
    prev_lw = None
    prev_rw = None
    feat_dim = compute_feat_dim()
    
    # 쿨다운 관리
    last_decision_time = 0
    stable_idle_count = 0
    last_result = None
    try:
        import time
        while True:
            # 프레임 수신 (base64 인코딩된 이미지)
            data = await websocket.receive_json()

            if "frame" not in data:
                continue

            # Base64 디코딩
            img_data = base64.b64decode(data["frame"])
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if frame is None:
                continue

            # RGB 변환 (Mediapipe용)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Mediapipe 처리
            hands_results = mp_hands_instance.process(rgb)
            pose_results = mp_pose_instance.process(rgb)

            # 피처 추출
            feat, vmag, prev_lw, prev_rw = extract_feature_and_speed(
                hands_results, pose_results, prev_lw, prev_rw, feat_dim
            )

            current_time = time.time()

            # 상태머신
            response = {"state": state, "vmag": vmag}
            
            # 마지막 결과 유지
            if last_result:
                response["last_result"] = last_result

            if state == "cooldown":
                # 쿨다운 중: 손이 내려가고 안정화될 때까지 대기
                time_since_decision = current_time - last_decision_time
                
                if vmag < IDLE_VEL_THR:
                    stable_idle_count += 1
                else:
                    stable_idle_count = 0
                
                response["cooldown_remaining"] = max(0, COOLDOWN_TIME - time_since_decision)
                response["stable_count"] = f"{stable_idle_count}/{STABLE_IDLE_COUNT}"
                
                # 쿨다운 완료 조건
                if time_since_decision >= COOLDOWN_TIME and stable_idle_count >= STABLE_IDLE_COUNT:
                    state = "idle"
                    stable_idle_count = 0
                    response["state"] = "idle"
                    response["message"] = "대기 완료"
                    print("🔄 쿨다운 완료 -> idle")
            
            elif state == "idle":
                if vmag >= IDLE_VEL_THR:
                    state = "active"
                    active_feats = [feat]
                    idle_count = 0
                    response["state"] = "active"
                    response["message"] = "수어 인식 시작"
                    print(f"🟢 Active 시작: vmag={vmag:.4f}")

            elif state == "active":
                active_feats.append(feat)

                if vmag < IDLE_VEL_THR:
                    idle_count += 1
                else:
                    idle_count = 0

                response["active_frames"] = len(active_feats)

                # 연속 idle 프레임으로 수어 종료
                if idle_count >= IDLE_END_N:
                    if len(active_feats) >= MIN_ACTIVE_FRAMES:
                        seq = np.stack(active_feats, axis=0)
                        L = seq.shape[0]
                        idxs = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
                        seq_fix = seq[idxs]
                        x = seq_fix.reshape(1, -1)

                        probs = clf.predict_proba(x)[0]
                        pred_id = int(np.argmax(probs))
                        label = id2label[pred_id]
                        confidence = float(probs[pred_id])

                        # 신뢰도 필터링
                        if confidence >= MIN_CONFIDENCE:
                            result = {
                                "text": label,
                            }
                            response["result"] = result
                            last_result = result
                            last_decision_time = current_time
                            state = "cooldown"
                            stable_idle_count = 0
                            response["state"] = "cooldown"
                            response["low_confidence"] = False
                            print(f"✅ 인식 완료: {label} ({confidence:.3f}) -> 쿨다운")
                        else:
                            # 낮은 신뢰도 결과도 전송하되 low_confidence 플래그 포함
                            result = {
                                "text": label,
                            }
                            response["result"] = result
                            response["low_confidence"] = True
                            response["message"] = f"낮은 신뢰도: {label} ({confidence:.3f})"
                            state = "idle"
                            response["state"] = "idle"
                            print(f"⚠️ 낮은 신뢰도: {label} ({confidence:.3f})")
                    else:
                        response["message"] = f"프레임 부족 ({len(active_feats)}/{MIN_ACTIVE_FRAMES})"
                        state = "idle"
                        response["state"] = "idle"
                        print(f"⚠️ 프레임 수 부족")

                    # active 데이터 초기화
                    active_feats = []
                    idle_count = 0
                    
            # 응답 전송
            await websocket.send_json(response)

    except WebSocketDisconnect:
        print("🔌 WebSocket 연결 종료")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")
        await websocket.close()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
