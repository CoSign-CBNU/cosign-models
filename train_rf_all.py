# train_rf_all.py
# random forest사용
# AI hub 제공 5개 영상 + 노트북 카메라 촬영 영상으로 라벨 학습
# 5-fold로 전체 데이터셋 교차검증(tr/ts 분리 X)
import os
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# 학습용/테스트용 데이터 디렉터리
TRAIN_DATASET_DIR = "./dataset_out"
TEST_DATASET_DIR  = "./dataset_out_test"

def load_all_data():
    # 1) train 쪽 X, y 로드
    X_train = np.load(os.path.join(TRAIN_DATASET_DIR, "X_seq.npy"))  # (25, T, F)
    y_train = np.load(os.path.join(TRAIN_DATASET_DIR, "y.npy"))      # (25,)

    # 2) test 쪽 X, y 로드
    X_test  = np.load(os.path.join(TEST_DATASET_DIR, "X_seq.npy"))   # (25, T, F)
    y_test  = np.load(os.path.join(TEST_DATASET_DIR, "y.npy"))       # (25,)

    # 3) 합치기
    X_all = np.concatenate([X_train, X_test], axis=0)  # (50, T, F)
    y_all = np.concatenate([y_train, y_test], axis=0)  # (50,)

    # 4) label_map은 train 쪽 것을 기준으로 사용
    label_map_path = os.path.join(TRAIN_DATASET_DIR, "label_map.json")
    with open(label_map_path, "r", encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    print("=== Loaded ALL dataset (train + test) ===")
    print(f"X_train: {X_train.shape}, X_test: {X_test.shape}")
    print(f"X_all : {X_all.shape}  (N, SEQ_LEN, FEAT_DIM)")
    print(f"y_all : {y_all.shape}  (N,)")
    print(f"labels: {lm['label2id']}")

    return X_all, y_all, id2label

def main():
    X, y, id2label = load_all_data()

    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)  # (N, T*F)

    print("\n=== 5-Fold Cross Validation on ALL 50 samples ===")
    print(f"Total samples: {N}")

    # 랜덤포레스트 정의
    base_clf = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        n_jobs=-1,
        random_state=42,
        class_weight=None,
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    label_ids    = sorted(id2label.keys())
    target_names = [id2label[i] for i in label_ids]

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/5")
        print(f"{'='*60}")

        X_train, X_val = X_flat[train_idx], X_flat[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight=None,
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc  = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average='macro', zero_division=0)
        rec  = recall_score(y_val, y_pred, average='macro', zero_division=0)
        f1   = f1_score(y_val, y_pred, average='macro', zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

        print(f"Val samples: {len(y_val)}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

        print(f"\nClassification Report (Fold {fold_idx}):")
        print(classification_report(
            y_val, y_pred,
            labels=label_ids,
            target_names=target_names,
            zero_division=0
        ))

        print(f"\nConfusion Matrix (Fold {fold_idx}) - 행=실제, 열=예측:")
        cm = confusion_matrix(y_val, y_pred, labels=label_ids)
        print(cm)

    # ================================
    # 교차검증 요약
    # ================================
    fold_accuracies = np.array(fold_accuracies)
    fold_precisions = np.array(fold_precisions)
    fold_recalls    = np.array(fold_recalls)
    fold_f1s        = np.array(fold_f1s)

    print(f"\n{'='*60}")
    print("=== Cross Validation Summary (ALL 50 samples) ===")
    print(f"{'='*60}")
    print(f"Accuracy per fold : {fold_accuracies}")
    print(f"Mean Accuracy     : {fold_accuracies.mean():.4f} (+/- {fold_accuracies.std():.4f})")
    print(f"Mean Precision    : {fold_precisions.mean():.4f} (+/- {fold_precisions.std():.4f})")
    print(f"Mean Recall       : {fold_recalls.mean():.4f} (+/- {fold_recalls.std():.4f})")
    print(f"Mean F1-Score     : {fold_f1s.mean():.4f} (+/- {fold_f1s.std():.4f})")

    # (선택) 전체 50개로 최종 모델 학습 + 저장
    print("\n=== Train final model on ALL 50 samples & save ===")
    final_clf = base_clf
    final_clf.fit(X_flat, y)

    model_path = os.path.join(TRAIN_DATASET_DIR, "rf_model_all.pkl")
    joblib.dump(final_clf, model_path)
    print(f"✅ Final model saved to: {model_path}")

if __name__ == "__main__":
    main()
