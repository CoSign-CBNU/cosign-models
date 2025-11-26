# extract_seq_test.py
# 테스트 셋용 키포인트 추출 (설정값 - 경로만 다름)
import os
import re
import cv2
import json
import numpy as np
from tqdm import tqdm
import mediapipe as mp

# ====== 설정 ======
DATA_DIR = r"./data_test"         # 입력 비디오 루트
OUT_DIR  = r"./dataset_out_test"  # 출력 루트
SEQ_LEN  = 60               # ✅ 영상당 고정 샘플 프레임 수 (60으로 변경)
ANGLE_NAMES = ["D", "F", "L", "R", "U"]  # 방향 키워드

# 피처 구성 플래그
MAX_HANDS        = 2     # 양손 처리(1 또는 2)
USE_BODY         = True  # 상체 피처 사용
USE_ELBOWS       = True  # 팔꿈치 포함
USE_HIPS         = True  # 엉덩이 포함
USE_LIMB_ANGLES  = True  # 팔 벡터/각도 피처 (상완/전완 + 각도)
USE_VELOCITY     = False # 손목 속도(xy) 피처로 추가할지 여부 (Idle 판별에는 내부적으로 사용)

# ✅ Idle 구간 판별용 파라미터
IDLE_VEL_THR      = 0.012  # 손목 속도 임계값 (이거보다 작으면 '거의 안 움직임')
MIN_ACTIVE_FRAMES = 5      # 실제 수어 구간 최소 프레임 수

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

def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

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

# --- 보조 수학 ---
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
    # 속도: 좌/우 손목 xy → 4 (지금은 USE_VELOCITY=False)
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

# ====== Idle 트리밍 & 리샘플링 유틸 ======
def trim_idle(seq, speeds, vel_thr=IDLE_VEL_THR, min_active_frames=MIN_ACTIVE_FRAMES, return_indices=False):
    """
    앞/뒤에서 손목 속도가 일정 임계값 이하인 구간(Idle)을 잘라내고,
    중간의 '실제 수어 구간'만 남긴다.

    seq    : [feat_vec_0, feat_vec_1, ...]  (list of np.array)
    speeds : [v0, v1, v2, ...]              (각 프레임의 손목 속도 스칼라)
    return_indices=True 이면 (잘린_seq, start, end) 튜플로 반환
    """
    n = len(seq)
    if n == 0 or len(speeds) != n:
        if return_indices:
            return seq, 0, n-1
        return seq

    # 앞에서부터 active(vel >= thr) 첫 위치 찾기
    start = 0
    while start < n and speeds[start] < vel_thr:
        start += 1

    # 뒤에서부터 active(vel >= thr) 마지막 위치 찾기
    end = n - 1
    while end >= 0 and speeds[end] < vel_thr:
        end -= 1

    # 전부 idle이거나, 너무 짧으면 그냥 원본 사용
    if end <= start or (end - start + 1) < min_active_frames:
        if return_indices:
            return seq, 0, n-1
        return seq

    # 여유 margin
    start = max(0, start - 1)
    end   = min(n - 1, end + 1)

    trimmed = seq[start:end+1]

    if return_indices:
        return trimmed, start, end
    return trimmed


def resample_seq(seq, target_len, feat_dim):
    """
    seq: list of feature vectors (길이: variable)
    target_len: 최종 SEQ_LEN (예: 120)
    """
    if len(seq) == 0:
        return np.zeros((target_len, feat_dim), np.float32)

    n = len(seq)
    arr = np.stack(seq, axis=0)  # (n, feat_dim)

    if n == target_len:
        return arr
    # n != target_len 이면 균등 샘플링
    idxs = np.linspace(0, n - 1, target_len, dtype=int)
    return arr[idxs]

# ====== 비디오 처리 ======
def process_video(video_path: str, seq_len: int = SEQ_LEN) -> np.ndarray:
    """
    비디오 전체 프레임에서 손+몸 피처 추출
    1) 모든 프레임에 대해 feature + 손목 속도 계산
    2) 앞/뒤 idle(손 거의 안 움직이는 구간) 잘라냄
    3) 남은 구간에서 seq_len(예: 120)개 프레임을 균등 샘플링
       → (seq_len, feat_dim)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feat_dim = compute_feat_dim()
    seq_full = []   # 모든 프레임의 feature
    speeds   = []   # 각 프레임의 손목 속도 (정규화된 xy 기준)

    prev_lw = None
    prev_rw = None

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose_det, \
         mp_hands.Hands(max_num_hands=MAX_HANDS,
                        min_detection_confidence=0.5,
                        min_tracking_confidence=0.5) as hands_det:

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pose_res = pose_det.process(rgb) if USE_BODY else None
            pose33 = pose_to_arr(pose_res.pose_landmarks) if pose_res else np.zeros((33, 3), np.float32)

            hands_res = hands_det.process(rgb)
            left_rel = np.zeros((21, 3), np.float32)
            right_rel = np.zeros((21, 3), np.float32)
            lw_abs = None
            rw_abs = None

            if hands_res.multi_hand_landmarks and hands_res.multi_handedness:
                for hlm, handed in zip(hands_res.multi_hand_landmarks, hands_res.multi_handedness):
                    lr = handed.classification[0].label  # 'Left'/'Right'
                    rel = lm21_to_rel(hlm)
                    wrist_abs = np.array(
                        [hlm.landmark[0].x, hlm.landmark[0].y, hlm.landmark[0].z],
                        np.float32
                    )
                    if lr == 'Left':
                        left_rel = rel
                        lw_abs = wrist_abs
                    else:
                        right_rel = rel
                        rw_abs = wrist_abs

            # 손 피처 (상대좌표)
            if MAX_HANDS == 2:
                hand_feat = np.concatenate(
                    [left_rel.reshape(-1), right_rel.reshape(-1)],
                    axis=0
                )
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

            # 손목 속도 계산 (Idle 판별용)
            lv = np.zeros(2, np.float32)
            rv = np.zeros(2, np.float32)
            if lw_abs is not None and prev_lw is not None:
                lv = (lw_abs[:2] - prev_lw[:2]) / max(norm_scale, 1e-6)
            if rw_abs is not None and prev_rw is not None:
                rv = (rw_abs[:2] - prev_rw[:2]) / max(norm_scale, 1e-6)

            # 속도 스칼라: 두 손 중 더 큰 값 사용
            vmag = max(np.linalg.norm(lv), np.linalg.norm(rv))
            speeds.append(float(vmag))

            # (옵션) 속도를 피처로도 넣고 싶으면 USE_VELOCITY 플래그로 제어
            if USE_VELOCITY:
                feat_parts.append(np.concatenate([lv, rv]).astype(np.float32))

            feat = np.concatenate(feat_parts, axis=0).astype(np.float32)

            # 안전장치: 계산된 길이가 예상과 다르면 zero-pad/trim
            if feat.shape[0] != feat_dim:
                if feat.shape[0] < feat_dim:
                    pad = np.zeros(feat_dim - feat.shape[0], np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                else:
                    feat = feat[:feat_dim]

            seq_full.append(feat)
            prev_lw, prev_rw = lw_abs, rw_abs

    cap.release()

    # 프레임이 하나도 없으면 그냥 zero로
    if len(seq_full) == 0:
        return np.zeros((seq_len, feat_dim), np.float32)

    # ✅ 1) Idle 구간 잘라내기
    seq_trim = trim_idle(seq_full, speeds)

    # ✅ 2) 잘린 구간에서 다시 seq_len 프레임으로 리샘플링
    seq_final = resample_seq(seq_trim, seq_len, feat_dim)

    return seq_final

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

        videos = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith((".mp4", ".mov", ".avi", ".mkv"))
        ]

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
            # ✅ 실제 길이(seq.shape[0]) 기준으로 저장
            np.savetxt(csv_path, seq.reshape(seq.shape[0], -1), delimiter=",")

            meta.append({
                "label": label,
                "angle": angle,
                "video": vpath,
                "npy": npy_path,
                "csv": csv_path,
                "seq_len": int(seq.shape[0]),
                "feat_dim": int(seq.shape[1]),
                "flags": {
                    "MAX_HANDS": MAX_HANDS,
                    "USE_BODY": USE_BODY,
                    "USE_ELBOWS": USE_ELBOWS,
                    "USE_HIPS": USE_HIPS,
                    "USE_LIMB_ANGLES": USE_LIMB_ANGLES,
                    "USE_VELOCITY": USE_VELOCITY,
                    "IDLE_VEL_THR": IDLE_VEL_THR,
                    "MIN_ACTIVE_FRAMES": MIN_ACTIVE_FRAMES,
                }
            })

    with open(os.path.join(OUT_DIR, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Done. Saved to: {OUT_DIR}")
    print(f"Total items: {len(meta)}")

if __name__ == "__main__":
    main()
