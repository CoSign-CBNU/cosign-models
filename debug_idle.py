# debug_idle.py
# 영상의 idle 영역과 active 영역 탐지 및 시각화
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

from extract_keypoints import (
    mp_pose, mp_hands,
    MAX_HANDS, USE_BODY, USE_LIMB_ANGLES,
    IDLE_VEL_THR, SEQ_LEN,
    compute_feat_dim,
    lm21_to_rel,
    pose_to_arr,
    _body_rel_features_full,
    _limb_features,
    trim_idle,
)

def debug_visualize_idle(video_path: str):
    """
    1) 영상 전체 프레임에 대해 feature + 손목 속도 계산
    2) trim_idle로 앞/뒤 idle 구간 자름 (start, end 인덱스 얻음)
    3) 잘린 구간에서 SEQ_LEN개 리샘플링할 때 사용되는 인덱스도 계산
    4) 속도 그래프 위에 모든 정보 시각화
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    feat_dim = compute_feat_dim()
    seq_full = []
    speeds   = []

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
                    lr = handed.classification[0].label  # 'Left' or 'Right'
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

            # 손 피처 (원래대로)
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

            vmag = max(np.linalg.norm(lv), np.linalg.norm(rv))
            speeds.append(float(vmag))

            # feat_dim 맞춰서 저장 (process_video와 동일)
            feat = np.concatenate(feat_parts, axis=0).astype(np.float32)
            if feat.shape[0] != feat_dim:
                if feat.shape[0] < feat_dim:
                    pad = np.zeros(feat_dim - feat.shape[0], np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                else:
                    feat = feat[:feat_dim]

            seq_full.append(feat)
            prev_lw, prev_rw = lw_abs, rw_abs

    cap.release()

    n = len(seq_full)
    if n == 0:
        print("⚠ 프레임이 없습니다.")
        return

    # 1) Idle 트리밍 (start/end 인덱스 얻기)
    seq_trim, start, end = trim_idle(seq_full, speeds, return_indices=True)
    print(f"전체 프레임 수: {n}")
    print(f"활성 구간: [{start} ~ {end}] (길이: {end - start + 1})")

    # 2) 리샘플링 시 사용될 인덱스 (활성 구간 내부 기준)
    active_len = end - start + 1
    if active_len <= 0:
        print("⚠ 활성 구간이 유효하지 않아 리샘플링 생략.")
        sample_idxs = np.array([], dtype=int)
    else:
        local_idxs = np.linspace(0, active_len - 1, SEQ_LEN, dtype=int)  # 활성구간 내부 인덱스
        sample_idxs = start + local_idxs  # 전체 프레임 기준 인덱스

    # 3) 시각화
    frames = np.arange(n)

    plt.figure(figsize=(12, 5))
    plt.plot(frames, speeds, label="wrist speed")
    plt.axhline(IDLE_VEL_THR, color="red", linestyle="--", label="IDLE_VEL_THR")

    # idle 앞/뒤 영역 채우기
    if start > 0:
        plt.axvspan(0, start, color="gray", alpha=0.2, label="idle (start)")
    if end < n - 1:
        plt.axvspan(end, n - 1, color="gray", alpha=0.2, label="idle (end)")

    # 활성 구간 강조
    plt.axvspan(start, end, color="green", alpha=0.1, label="active region")

    # 샘플링되는 프레임 표시 (120개)
    if len(sample_idxs) > 0:
        plt.scatter(sample_idxs, np.array(speeds)[sample_idxs], color="black", s=20, label="resampled frames")

    plt.xlabel("Frame index")
    plt.ylabel("Wrist speed (normalized)")
    plt.title(f"Idle trimming debug: {os.path.basename(video_path)}")
    plt.legend(loc="upper right")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_video = r"data/1/NIA_SL_WORD1501_REAL01_F.mp4"  # 파악할 영상 경로
    debug_visualize_idle(test_video)
