# train_rf.py
# Random Forest 사용
# AI Hub 제공 영상으로 라벨 학습
# 5-fold 교차검증(tr/ts 분리 X)

import os
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ===== 경로 설정 =====
DATASET_DIR = "./dataset_out"
X_PATH = os.path.join(DATASET_DIR, "X_seq.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")


# ===== 데이터 로드 =====
def load_data():
    # X: (N, SEQ_LEN, FEAT_DIM)
    X = np.load(X_PATH)       # float32일 가능성 큼
    y = np.load(Y_PATH)       # int64

    # label_map 로드 (id2label → 정수 키로 변환)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)

    label2id = lm["label2id"]                    # 예: {"운전면허": 0, ...}
    id2label = {int(k): v for k, v in lm["id2label"].items()}  # 예: {0: "운전면허", ...}

    print("=== Loaded dataset ===")
    print(f"X shape: {X.shape}  (N, SEQ_LEN, FEAT_DIM)")
    print(f"y shape: {y.shape}  (N,)")
    print(f"labels : {label2id}")
    return X, y, id2label


# ===== 모델 생성 함수 =====
def create_model():
    return RandomForestClassifier(
        n_estimators=200,      # 트리 개수
        max_depth=None,        # 필요하면 20 등으로 제한 가능
        max_features="log2",
        min_samples_leaf=2,
        min_samples_split=10,
        n_jobs=-1,             # CPU 전부 사용
        random_state=42,
        class_weight=None      # 불균형 심해지면 "balanced" 고려
    )


# ===== 메인 =====
def main():
    X, y, id2label = load_data()

    N, T, F = X.shape  # N=샘플 수, T=시퀀스 길이, F=피처 차원
    # 랜덤포레스트는 2D 입력(X_samples, n_features)을 기대 → flatten
    X_flat = X.reshape(N, T * F)

    print("\n=== 5-Fold Cross Validation ===")
    print(f"Total samples: {N}")

    # 5-Fold 교차검증 (stratified로 라벨 비율 유지)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    target_names = [id2label[i] for i in sorted(id2label.keys())]

    # ----- Fold별 학습 & 평가 -----
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}/5")
        print(f"{'=' * 60}")

        # Train/Test split
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # 모델 생성 & 학습
        fold_clf = create_model()
        fold_clf.fit(X_train, y_train)

        # 예측
        y_pred = fold_clf.predict(X_test)

        # 메트릭 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

        print(f"Test samples: {len(y_test)}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

        print(f"\nClassification Report (Fold {fold_idx}):")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=target_names,
                zero_division=0,
            )
        )

        print(f"\nConfusion Matrix (Fold {fold_idx}) - 행=실제, 열=예측:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)

    # ----- 교차검증 요약 -----
    print(f"\n{'=' * 60}")
    print("=== Cross Validation Summary ===")
    print(f"{'=' * 60}")
    fold_accuracies = np.array(fold_accuracies)
    fold_precisions = np.array(fold_precisions)
    fold_recalls = np.array(fold_recalls)
    fold_f1s = np.array(fold_f1s)

    print(f"Accuracy per fold:  {fold_accuracies}")
    print(
        f"Mean Accuracy:      {fold_accuracies.mean():.4f} (+/- {fold_accuracies.std():.4f})"
    )
    print(
        f"Mean Precision:     {fold_precisions.mean():.4f} (+/- {fold_precisions.std():.4f})"
    )
    print(
        f"Mean Recall:        {fold_recalls.mean():.4f} (+/- {fold_recalls.std():.4f})"
    )
    print(
        f"Mean F1-Score:      {fold_f1s.mean():.4f} (+/- {fold_f1s.std():.4f})"
    )

    # ----- 전체 데이터로 최종 모델 학습 & 저장 -----
    print("\n=== Training final model on full dataset ===")
    final_clf = create_model()
    final_clf.fit(X_flat, y)

    # 모델 저장 (추론용)
    model_path = os.path.join(DATASET_DIR, "rf_model.pkl")
    joblib.dump(final_clf, model_path)
    print(f"\n✅ Model saved to: {model_path}")


if __name__ == "__main__":
    main()
