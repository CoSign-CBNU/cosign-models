# eval_rf_on_new.py
# 새로 찍은 test 영상 rf 모델 정확도 파악
import os
import json
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

TRAIN_DATASET_DIR = "./dataset_out"       # 학습 데이터 + 모델
TEST_DATASET_DIR  = "./dataset_out_test"  # 방금 만든 테스트셋

# ------------------------------
# 1. train/test 특징을 다 로드
# ------------------------------
X_train = np.load(os.path.join(TRAIN_DATASET_DIR, "X_seq.npy"))
y_train = np.load(os.path.join(TRAIN_DATASET_DIR, "y.npy"))

X_test  = np.load(os.path.join(TEST_DATASET_DIR, "X_seq.npy"))
y_test  = np.load(os.path.join(TEST_DATASET_DIR, "y.npy"))

# ------------------------------
# 2. train/test 데이터가 완전히 같은 게 있는지 검사
# ------------------------------
same = np.any([
    np.allclose(X_train[i], X_test[j])
    for i in range(len(X_train))
    for j in range(len(X_test))
])
print("train과 test에 완전히 동일한 시퀀스가 있는가?", same)
print("---------------------------------------------------------")

# 1) label_map, 모델 로드
label_map_path = os.path.join(TRAIN_DATASET_DIR, "label_map.json")
model_path     = os.path.join(TRAIN_DATASET_DIR, "rf_model.pkl")

with open(label_map_path, "r", encoding="utf-8") as f:
    lm = json.load(f)
id2label = {int(k): v for k, v in lm["id2label"].items()}
label_ids = sorted(id2label.keys())
target_names = [id2label[i] for i in label_ids]

clf = joblib.load(model_path)

# 2) 테스트셋 로드
X_test = np.load(os.path.join(TEST_DATASET_DIR, "X_seq.npy"))  # (M, T, F)
y_test = np.load(os.path.join(TEST_DATASET_DIR, "y.npy"))      # (M,)

M, T, F = X_test.shape
X_test_flat = X_test.reshape(M, T * F)

print("=== New Test Set ===")
print("X_test:", X_test.shape, "y_test:", y_test.shape)

# 3) 예측 + 평가
y_pred = clf.predict(X_test_flat)

acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy on new test videos: {acc * 100:.2f}%")

print("\nClassification Report on new test videos:")
print(classification_report(
    y_test, y_pred,
    labels=label_ids,
    target_names=target_names,
    zero_division=0
))

print("\nConfusion Matrix on new test videos (행=실제, 열=예측):")
print(confusion_matrix(y_test, y_pred, labels=label_ids))
