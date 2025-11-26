# build_test_repr.py
# build_dataset 에서 만들어진 라벨 정의를 재사용해서 Test 세트만 만듦
import os
import json
import numpy as np

TRAIN_DATASET_DIR = "./dataset_out"       # 학습 때 쓰던 폴더
TEST_DATASET_DIR  = "./dataset_out_test"  # 방금 추출한 테스트 특징 폴더

X_OUT_PATH = os.path.join(TEST_DATASET_DIR, "X_seq.npy")
Y_OUT_PATH = os.path.join(TEST_DATASET_DIR, "y.npy")

def main():
    # 1) 학습 때의 label_map 재사용 (라벨 ↔ id 매핑 통일)
    label_map_path = os.path.join(TRAIN_DATASET_DIR, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    label2id = lm["label2id"]   # 예: {"1":0, "2":1, ...}
    
    # 한글 라벨 정보 (있으면)
    word_mapping = lm.get("word_mapping", {})
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    X_list = []
    y_list = []

    # 2) 테스트 특징 npy 모으기
    for label_name in sorted(os.listdir(TEST_DATASET_DIR)):
        label_dir = os.path.join(TEST_DATASET_DIR, label_name)
        if not os.path.isdir(label_dir):
            continue
        if label_name not in label2id:
            print(f"⚠️ label '{label_name}' not in train label2id, skip")
            continue

        label_id = label2id[label_name]

        for fn in sorted(os.listdir(label_dir)):
            if not fn.lower().endswith(".npy"):
                continue
            path = os.path.join(label_dir, fn)
            arr = np.load(path)  # (T, F)
            if arr.ndim != 2:
                print(f"⚠️ skip {path}, shape={arr.shape}")
                continue
            X_list.append(arr)
            y_list.append(label_id)

    X = np.stack(X_list, axis=0)   # (M, T, F)
    y = np.array(y_list, dtype=int)

    print("=== Built test set ===")
    print("X:", X.shape)
    print("y:", y.shape)
    print("label2id:", label2id)
    print("id2label (한글):", id2label)

    np.save(X_OUT_PATH, X)
    np.save(Y_OUT_PATH, y)
    print(f"✅ Saved X to {X_OUT_PATH}")
    print(f"✅ Saved y to {Y_OUT_PATH}")

if __name__ == "__main__":
    main()
