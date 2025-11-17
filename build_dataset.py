# build_dataset.py
# X_seq 와 Y(라벨=폴더 이름)으로 나눔
import os
import json
import numpy as np

DATASET_DIR = "./dataset_out"   # extract_keypoints.py가 만든 폴더
OUT_X       = "X_seq.npy"       # (N, 120, feat_dim)
OUT_Y       = "y.npy"           # (N,)
OUT_LABELS  = "label_map.json"  # 라벨 <-> id 매핑

def main():
    label_dirs = [
        d for d in os.listdir(DATASET_DIR)
        if os.path.isdir(os.path.join(DATASET_DIR, d))
    ]
    label_dirs = sorted(label_dirs)   # 예: ["1", "2", "3", "4", "5"]

    if not label_dirs:
        print("⚠ label 폴더가 없습니다. dataset_out 구조를 확인하세요.")
        return

    # 폴더 이름을 라벨 id로 매핑
    label2id = {label: idx for idx, label in enumerate(label_dirs)}
    id2label = {idx: label for label, idx in label2id.items()}

    X_list = []
    y_list = []
    paths  = []

    for label in label_dirs:
        label_id = label2id[label]
        label_dir = os.path.join(DATASET_DIR, label)

        npy_files = [
            f for f in os.listdir(label_dir)
            if f.lower().endswith(".npy")
        ]

        if not npy_files:
            print(f"⚠ 폴더 {label_dir} 안에 .npy 파일이 없습니다. 건너뜁니다.")
            continue

        for fn in sorted(npy_files):
            path = os.path.join(label_dir, fn)
            arr = np.load(path)

            if arr.ndim != 2:
                print(f"⚠ {path} : 예상과 다른 shape {arr.shape}, 건너뜁니다.")
                continue

            X_list.append(arr)        # (SEQ_LEN, feat_dim)
            y_list.append(label_id)   # 정수 라벨
            paths.append(path)

    if not X_list:
        print("⚠ 로드된 샘플이 없습니다. .npy 파일을 확인하세요.")
        return

    # (N, SEQ_LEN, feat_dim) 으로 스택
    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=np.int64)

    # 저장
    out_X_path = os.path.join(DATASET_DIR, OUT_X)
    out_Y_path = os.path.join(DATASET_DIR, OUT_Y)
    out_L_path = os.path.join(DATASET_DIR, OUT_LABELS)

    np.save(out_X_path, X)
    np.save(out_Y_path, y)

    with open(out_L_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "label2id": label2id,
                "id2label": {str(k): v for k, v in id2label.items()},
                "n_samples": int(X.shape[0]),
                "seq_len": int(X.shape[1]),
                "feat_dim": int(X.shape[2]),
                "paths": paths,   # 원하면 나중에 추적용
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("✅ Dataset built!")
    print(f"  X shape: {X.shape}   (N, SEQ_LEN, feat_dim)")
    print(f"  y shape: {y.shape}   (N,)")
    print(f"  labels : {label2id}")
    print(f"  saved  : {out_X_path}, {out_Y_path}, {out_L_path}")

if __name__ == "__main__":
    main()
