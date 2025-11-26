# realtime_sign.py
import cv2
import numpy as np
import mediapipe as mp
import joblib
import json
import time
import sys
import io
from PIL import ImageFont, ImageDraw, Image

# Windows 콘솔 한글 출력 설정
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# ===== 설정 =====
SEQ_LEN         = 60   # 학습 때 쓰던 시퀀스 길이 (리샘플링 길이)
CAM_INDEX       = 0     # 웹캠 번호
MODEL_PATH      = "./dataset_out/rf_model_all.pkl"   # RF 모델 경로
LABEL_MAP_PATH  = "./dataset_out/label_map.json"     # label_map.json 경로

# idle/active 판별 파라미터 (extract_keypoints.py와 맞춤)
IDLE_VEL_THR      = 0.020  # 손목 속도 임계값
MIN_ACTIVE_FRAMES = 10    # 최소 active 프레임 수
IDLE_END_N        = 5      # 연속 idle 프레임 수 ≥ IDLE_END_N 이면 수어 종료로 판단

# ===== extract_keypoints.py와 동일한 피처 설정 =====
MAX_HANDS        = 2     # 양손 처리
USE_BODY         = True  # 상체 피처 사용
USE_ELBOWS       = True  # 팔꿈치 포함
USE_HIPS         = True  # 엉덩이 포함
USE_LIMB_ANGLES  = True  # 팔 벡터/각도 피처
USE_VELOCITY     = False # 손목 속도 피처 (현재는 사용 안 함)

# ===== 모델 & 라벨 매핑 로드 =====
def load_model(model_path, label_map_path):
    clf = joblib.load(model_path)

    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    # id2label에 한글 라벨이 저장되어 있음
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    return clf, id2label

# ===== Mediapipe 준비 =====
mp_hands = mp.solutions.hands
mp_pose  = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# ===== Pose 인덱스 정의 =====
POSE_NOSE = 0
POSE_LSHO, POSE_RSHO = 11, 12
POSE_LELB, POSE_RELB = 13, 14
POSE_LWR,  POSE_RWR  = 15, 16
POSE_LHIP, POSE_RHIP = 23, 24

# ===== 유틸 함수 =====
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
    count = 4  # nose, shoulder_mid, lsho, rsho
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
    """상체 기준 상대피처 (extract_keypoints.py와 동일)"""
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
    """팔 벡터/각도 피처 (extract_keypoints.py와 동일)"""
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
        elbow_cos    = _angle_cos(_unit(upper), _unit(lower))
        shoulder_cos = _angle_cos(_unit(trunk), _unit(upper))
        return np.concatenate([upper, lower,
                               np.array([elbow_cos, shoulder_cos], np.float32)])

    feats.append(arm_side(lsho, lelb, lwr))
    feats.append(arm_side(rsho, relb, rwr))

    return np.concatenate(feats).astype(np.float32)

# ===== 1프레임에서 피처 + 손목 속도 계산 =====
def extract_feature_and_speed(hands_results, pose_results,
                              prev_lw, prev_rw, feat_dim):
    """
    extract_keypoints.py 의 process_video() 한 프레임 분을
    실시간용으로 옮겨온 버전.

    반환:
        feat : (feat_dim,) float32
        vmag : 손목 속도 (idle/active 판별용 스칼라)
        lw_abs, rw_abs : 이번 프레임의 좌/우 손목 절대좌표 (다음 프레임용)
    """
    # Pose 처리
    if USE_BODY and pose_results and pose_results.pose_landmarks:
        pose33 = pose_to_arr(pose_results.pose_landmarks)
    else:
        pose33 = np.zeros((33, 3), np.float32)

    # Hand 처리
    left_rel  = np.zeros((21, 3), np.float32)
    right_rel = np.zeros((21, 3), np.float32)
    lw_abs = None
    rw_abs = None

    if hands_results and hands_results.multi_hand_landmarks and hands_results.multi_handedness:
        for hlm, handed in zip(hands_results.multi_hand_landmarks,
                               hands_results.multi_handedness):
            lr = handed.classification[0].label  # 'Left'/'Right'
            rel = lm21_to_rel(hlm)
            wrist_abs = np.array(
                [hlm.landmark[0].x, hlm.landmark[0].y, hlm.landmark[0].z],
                np.float32
            )
            if lr == "Left":
                left_rel = rel
                lw_abs = wrist_abs
            else:
                right_rel = rel
                rw_abs = wrist_abs

    # 손 피처 (상대좌표)
    if MAX_HANDS == 2:
        hand_feat = np.concatenate(
            [left_rel.reshape(-1), right_rel.reshape(-1)],
            axis=0,
        )
    else:
        if lw_abs is not None:
            hand_feat = left_rel.reshape(-1)
        else:
            hand_feat = right_rel.reshape(-1)

    feat_parts = [hand_feat]

    # 상체 상대 + 정규화 스케일
    norm_scale = 1.0
    if USE_BODY:
        body_rel, norm_scale = _body_rel_features_full(pose33, lw_abs, rw_abs)
        feat_parts.append(body_rel)

    # 팔 벡터/각도
    if USE_BODY and USE_LIMB_ANGLES:
        limb = _limb_features(pose33, lw_abs, rw_abs, norm_scale)
        feat_parts.append(limb)

    # 속도 피처는 현재 미사용 (USE_VELOCITY=False)

    feat = np.concatenate(feat_parts, axis=0).astype(np.float32)

    # 길이 보정
    if feat.shape[0] != feat_dim:
        if feat.shape[0] < feat_dim:
            pad = np.zeros(feat_dim - feat.shape[0], np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        else:
            feat = feat[:feat_dim]

    # 손목 속도 계산 (idle 판별용)
    lv = np.zeros(2, np.float32)
    rv = np.zeros(2, np.float32)
    if lw_abs is not None and prev_lw is not None:
        lv = (lw_abs[:2] - prev_lw[:2]) / max(norm_scale, 1e-6)
    if rw_abs is not None and prev_rw is not None:
        rv = (rw_abs[:2] - prev_rw[:2]) / max(norm_scale, 1e-6)

    vmag = max(np.linalg.norm(lv), np.linalg.norm(rv))

    return feat, float(vmag), lw_abs, rw_abs

# ===== 메인 =====
# ===== 한글 텍스트 표시 함수 =====
def put_korean_text(img, text, pos, font_size=25, color=(255, 255, 255)):
    """PIL을 사용하여 한글 텍스트를 이미지에 표시"""
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)
    
    # Windows 기본 한글 폰트 사용
    try:
        font = ImageFont.truetype("malgun.ttf", font_size)  # 맑은 고딕
    except:
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/malgun.ttf", font_size)
        except:
            font = ImageFont.load_default()
    
    draw.text(pos, text, font=font, fill=color[::-1])  # BGR to RGB
    img_result = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    return img_result


def main():
    clf, id2label = load_model(MODEL_PATH, LABEL_MAP_PATH)
    print("✅ 모델, 라벨 매핑 로드 완료")

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print("❌ 카메라를 열 수 없습니다.")
        return

    # 해상도/FPS 설정 시도
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 실측 FPS 한 번 찍어보기 (1초)
    start = time.time()
    frames = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames += 1
        if time.time() - start >= 1.0:
            break
    print(f"📸 실측 웹캠 FPS ≈ {frames} fps")

    # Mediapipe 인스턴스
    hands = mp_hands.Hands(
        max_num_hands=MAX_HANDS,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    pose = mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    feat_dim = compute_feat_dim()

    # idle/active 상태 관리용 변수들
    state = "idle"   # "idle" 또는 "active"
    active_feats = []
    idle_count = 0
    prev_lw = None
    prev_rw = None

    last_label = None
    last_prob  = 0.0

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("프레임을 읽을 수 없습니다. 종료합니다.")
            break

        # Mediapipe는 원본(반전 안 한 영상)으로 처리
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        hands_results = hands.process(rgb)
        pose_results  = pose.process(rgb)
        
        # 사용자에게 보여주는 화면만 반전
        frame = cv2.flip(frame, 1)

        # 랜드마크 그리기(디버그용)
        if hands_results and hands_results.multi_hand_landmarks:
            for hand_lm in hands_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_lm, mp_hands.HAND_CONNECTIONS
                )
        if pose_results and pose_results.pose_landmarks:
            mp_drawing.draw_landmarks(
                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS
            )

        # 1프레임 → feature + 손목 속도
        feat, vmag, prev_lw, prev_rw = extract_feature_and_speed(
            hands_results, pose_results, prev_lw, prev_rw, feat_dim
        )

        # ===== idle / active 상태머신 =====
        if state == "idle":
            if vmag >= IDLE_VEL_THR:
                # 수어 시작
                state = "active"
                active_feats = [feat]
                idle_count = 0
        elif state == "active":
            active_feats.append(feat)

            if vmag < IDLE_VEL_THR:
                idle_count += 1
            else:
                idle_count = 0

            # 연속 idle 프레임이 충분히 나오면 수어 종료로 판단
            if idle_count >= IDLE_END_N:
                if len(active_feats) >= MIN_ACTIVE_FRAMES:
                    seq = np.stack(active_feats, axis=0)  # (L, feat_dim)
                    L = seq.shape[0]
                    idxs = np.linspace(0, L - 1, SEQ_LEN, dtype=int)
                    seq_fix = seq[idxs]                  # (60, feat_dim)
                    x = seq_fix.reshape(1, -1)           # (1, 60*feat_dim)

                    probs = clf.predict_proba(x)[0]
                    pred_id = int(np.argmax(probs))
                    last_label = id2label[pred_id]
                    last_prob  = float(probs[pred_id])

                    print(f"✅ DECISION: {last_label} ({last_prob:.3f})")

                # 상태 초기화
                state = "idle"
                active_feats = []
                idle_count = 0

        # ===== FPS 계산 =====
        now = time.time()
        fps = 1.0 / (now - prev_time)
        prev_time = now

        # ===== 화면 표시 (한글 지원) =====
        y0 = 30
        # 상태
        frame = put_korean_text(
            frame,
            f"상태: {state} (vmag={vmag:.3f})",
            (10, y0),
            font_size=20,
            color=(0, 255, 255)
        )
        y0 += 30

        # 마지막으로 결정된 라벨
        if last_label is not None:
            frame = put_korean_text(
                frame,
                f"인식 결과: {last_label} ({last_prob:.2f})",
                (10, y0),
                font_size=28,
                color=(0, 255, 0)
            )
            y0 += 40

        # FPS
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, h - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 0, 0),
            2,
        )

        cv2.imshow("Realtime Sign Classification (idle-trim style)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("종료합니다.")
            break

    cap.release()
    cv2.destroyAllWindows()
    hands.close()
    pose.close()

if __name__ == "__main__":
    main()
