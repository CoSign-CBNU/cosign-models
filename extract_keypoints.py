import os
import re
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ====== 설정 ======
DATA_DIR = r"./data"         # 입력 비디오 루트
OUT_DIR  = r"./dataset_out"  # 출력 루트
SEQ_LEN  = 30                # 영상당 고정 샘플 프레임 수
ANGLE_NAMES = ["D", "F", "L", "R", "U"]  # 방향 키워드

# 피처 구성 플래그
MAX_HANDS        = 2     # 양손 처리(1 또는 2)
USE_BODY         = True  # 상체 피처 사용
USE_ELBOWS       = True  # 팔꿈치 포함
USE_HIPS         = True  # 엉덩이 포함
USE_LIMB_ANGLES  = True  # 팔 벡터/각도 피처 (상완/전완 + 각도)
USE_VELOCITY     = False # 손목 속도(xy) 추가

# ====== Mediapipe 준비 ======
mp_hands = mp.solutions.hands
mp_pose = mp.solutions.pose

# ====== Pose 인덱스 정의 ======
POSE_NOSE = 0
POSE_LSHO, POSE_RSHO = 11, 12
POSE_LELB, POSE_RELB = 13, 14
POSE_LWR,  POSE_RWR  = 15, 16
POSE_LHIP, POSE_RHIP = 23, 24

# ====== 유틸 ======
def parse_angle(filename: str) -> str:
    """파일명에서 각도 추출 (예: *_L_*.mp4, *_U.mp4 등)"""
    name = filename.upper()
    for a in ANGLE_NAMES:
        if f"_{a}." in name or f"_{a}_" in name:
            return a
    return "unknown"

def sample_indices(n_frames: int, seq_len: int) -> np.ndarray:
    """전체 프레임 중 seq_len개 균등 샘플링 인덱스 반환"""
    if n_frames <= 0:
        return np.array([], dtype=int)
    return np.linspace(0, n_frames - 1, seq_len, dtype=int)

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

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

# --- 보조 수학 ---
def _unit(v):
    n = np.linalg.norm(v)
    if n < 1e-6: return v*0
    return v / n

def _angle_cos(u, v):
    nu, nv = np.linalg.norm(u), np.linalg.norm(v)
    if nu < 1e-6 or nv < 1e-6: return 0.0
    return float(np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0))

# ====== 피처 차원 계산 ======
def _count_body_targets():
    # nose, shoulder_mid, lsho, rsho
    count = 4
    if USE_ELBOWS:
        count += 2  # lelb, relb
    if USE_HIPS:
        count += 3  # lhip, rhip, hip_mid
    return count

def compute_feat_dim():
    # 손
    hand_dim = (21 * 3) * (2 if MAX_HANDS == 2 else 1)
    # 상체 상대 (target 개수 * 3) * (hand sides: 2 or 1)
    body_dim = 0
    if USE_BODY:
        body_dim = _count_body_targets() * 3 * (2 if MAX_HANDS == 2 else 1)
    # 팔 벡터/각도: 좌/우 각각 (상완2D=2 + 전완2D=2 + 각도2) = 6 → 양쪽 합 12
    limb_dim = 12 if (USE_BODY and USE_LIMB_ANGLES) else 0
    # 속도: 좌/우 손목 xy → 4
    vel_dim = 4 if USE_VELOCITY else 0
    return hand_dim + body_dim + limb_dim + vel_dim

# ====== 상체 피처 계산 ======
def _body_rel_features_full(pose33, left_wrist_abs, right_wrist_abs):
    """
    상체 기준 상대피처 (코, 어깨중점, 좌/우 어깨, (옵션) 좌/우 팔꿈치, (옵션) 좌/우 엉덩이 + 엉덩이중점)
    정규화: 어깨폭(shw)와 토르소(어깨중점-엉덩이중점)의 평균 길이
    반환: (rel_features, norm_scale)
    """
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
        vecs = [ (wrist_abs - t) / norm_scale for t in targets ]
        return np.concatenate(vecs, axis=0).astype(np.float32)

    if MAX_HANDS == 2:
        l = rel(left_wrist_abs)
        r = rel(right_wrist_abs)
        return np.concatenate([l, r], axis=0), norm_scale
    else:
        tgt = left_wrist_abs if left_wrist_abs is not None else right_wrist_abs
        return rel(tgt), norm_scale

def _limb_features(pose33, lwr, rwr, norm_scale):
    """
    팔 벡터/각도 피처:
    - 좌/우 상완벡터(어깨→팔꿈치) [xy 2], 전완벡터(팔꿈치→손목) [xy 2]
    - 팔꿈치 각도(상완 vs 전완, cos) 1, 어깨 각도(몸통축 vs 상완, cos) 1
    → 한쪽 팔당 6, 양쪽 합 12
    """
    lsho, rsho = pose33[POSE_LSHO][:2], pose33[POSE_RSHO][:2]
    lelb, relb = pose33[POSE_LELB][:2], pose33[POSE_RELB][:2]

    # 몸통축: 어깨중점→엉덩이중점 (없으면 수직)
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
        return np.concatenate([upper, lower, np.array([elbow_cos, shoulder_cos], np.float32)])

    # 좌/우 팔
    feats.append(arm_side(lsho, lelb, lwr))
    feats.append(arm_side(rsho, relb, rwr))

    return np.concatenate(feats).astype(np.float32)  # 12

# ====== 비디오 처리 ======
def process_video(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    """비디오에서 프레임별 손+몸 피처 추출 → (seq_len, feat_dim)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    idxs = sample_indices(n_frames, seq_len)

    feat_dim = compute_feat_dim()
    seq = []
    prev_lw = None; prev_rw = None  # 속도용

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_det, \
         mp_hands.Hands(max_num_hands=MAX_HANDS,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands_det:

        for fi in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
            ok, frame = cap.read()
            if not ok or frame is None:
                seq.append(np.zeros(feat_dim, np.float32))
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose_det.process(rgb) if USE_BODY else None
            pose33 = pose_to_arr(pose_res.pose_landmarks) if pose_res else np.zeros((33, 3), np.float32)

            hands_res = hands_det.process(rgb)
            left_rel = np.zeros((21, 3), np.float32)
            right_rel = np.zeros((21, 3), np.float32)
            lw_abs = None; rw_abs = None

            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                for hlm, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                    lr = handed.classification[0].label  # 'Left'/'Right'
                    rel = lm21_to_rel(hlm)
                    wrist_abs = np.array([hlm.landmark[0].x, hlm.landmark[0].y, hlm.landmark[0].z], np.float32)
                    if lr == 'Left':
                        left_rel = rel; lw_abs = wrist_abs
                    else:
                        right_rel = rel; rw_abs = wrist_abs

            # 손 피처 (상대좌표)
            if MAX_HANDS == 2:
                hand_feat = np.concatenate([left_rel.reshape(-1), right_rel.reshape(-1)], axis=0)  # 126
            else:
                hand_feat = left_rel.reshape(-1) if lw_abs is not None else right_rel.reshape(-1)

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

            # 손목 속도(옵션, xy만)
            if USE_VELOCITY:
                if lw_abs is not None and prev_lw is not None:
                    lv = (lw_abs[:2] - prev_lw[:2]) / max(norm_scale, 1e-6)
                else:
                    lv = np.zeros(2, np.float32)
                if rw_abs is not None and prev_rw is not None:
                    rv = (rw_abs[:2] - prev_rw[:2]) / max(norm_scale, 1e-6)
                else:
                    rv = np.zeros(2, np.float32)
                feat_parts.append(np.concatenate([lv, rv]).astype(np.float32))

            feat = np.concatenate(feat_parts, axis=0).astype(np.float32)

            # 안전장치: 계산된 길이가 예상과 다르면 zero-pad/trim
            if feat.shape[0] != feat_dim:
                if feat.shape[0] < feat_dim:
                    pad = np.zeros(feat_dim - feat.shape[0], np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                else:
                    feat = feat[:feat_dim]

            seq.append(feat)
            prev_lw, prev_rw = lw_abs, rw_abs

    cap.release()
    return np.stack(seq, axis=0) if len(seq) > 0 else np.zeros((seq_len, compute_feat_dim()), np.float32)

# ====== 메인 루프 ======
def main():
    ensure_dir(OUT_DIR)
    meta = []

    for label in sorted(os.listdir(DATA_DIR)):
        label_dir = os.path.join(DATA_DIR, label)
        if not os.path.isdir(label_dir):
            continue

        out_label_dir = os.path.join(OUT_DIR, label)
        ensure_dir(out_label_dir)

        videos = [f for f in os.listdir(label_dir) if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))]

        for vf in tqdm(videos, desc=f"[{label}]"):
            vpath = os.path.join(label_dir, vf)
            angle = parse_angle(vf)

            try:
                seq = process_video(vpath, SEQ_LEN)
            except Exception as e:
                print(f"!! Fail {vpath}: {e}")
                continue

            base = os.path.splitext(vf)[0]
            out_base = f"{base}_{angle}" if angle != "unknown" else base

            npy_path = os.path.join(out_label_dir, out_base + ".npy")
            csv_path = os.path.join(out_label_dir, out_base + ".csv")
            np.save(npy_path, seq)
            np.savetxt(csv_path, seq.reshape(SEQ_LEN, -1), delimiter=",")

            meta.append({
                "label": label,
                "angle": angle,
                "video": vpath,
                "npy": npy_path,
                "csv": csv_path,
                "seq_len": int(SEQ_LEN),
                "feat_dim": int(seq.shape[1]),
                "flags": {
                    "MAX_HANDS": MAX_HANDS,
                    "USE_BODY": USE_BODY,
                    "USE_ELBOWS": USE_ELBOWS,
                    "USE_HIPS": USE_HIPS,
                    "USE_LIMB_ANGLES": USE_LIMB_ANGLES,
                    "USE_VELOCITY": USE_VELOCITY,
                }
            })

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Saved to: {OUT_DIR}")
    print(f"Total items: {len(meta)}")

if __name__ == "__main__":
    main()
