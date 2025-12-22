# train_linearsvc.py

import os
import json
import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

DATASET_DIR = "./dataset_out"
X_PATH = os.path.join(DATASET_DIR, "X_seq.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")


def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    print("=== Loaded dataset ===")
    print(f"X shape: {X.shape}  (N, SEQ_LEN, FEAT_DIM)")
    print(f"y shape: {y.shape}  (N,)")
    print(f"labels : {lm['label2id']}")
    return X, y, id2label


def create_model():
    # 스케일링 + LinearSVC 조합
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearSVC(
            C=0.4,                 # 1.0보다 작게 → 규제 강화
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )


def main():
    X, y, id2label = load_data()
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)

    print("\n=== 5-Fold CV (LinearSVC) ===")
    print(f"Total samples: {N}")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []

    label_ids = sorted(id2label.keys())
    target_names = [id2label[i] for i in label_ids]

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), 1):
        print(f"\n{'=' * 60}")
        print(f"Fold {fold_idx}/5")
        print(f"{'=' * 60}")

        X_train, X_val = X_flat[train_idx], X_flat[test_idx]
        y_train, y_val = y[train_idx], y[test_idx]

        clf = create_model()
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_val, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_val, y_pred, average="macro", zero_division=0)

        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)

        print(f"Val samples: {len(y_val)}")
        print(f"Accuracy : {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall   : {rec:.4f}")
        print(f"F1-Score : {f1:.4f}")

        print("\nClassification Report:")
        print(
            classification_report(
                y_val,
                y_pred,
                labels=label_ids,
                target_names=target_names,
                zero_division=0,
            )
        )

        print("\nConfusion Matrix (행=실제, 열=예측):")
        print(confusion_matrix(y_val, y_pred, labels=label_ids))

    print(f"\n{'=' * 60}")
    print("=== CV Summary (LinearSVC) ===")
    print(f"{'=' * 60}")
    print("Accuracy per fold :", np.array(fold_accuracies))
    print("Mean Accuracy     :", np.mean(fold_accuracies))
    print("Mean Precision    :", np.mean(fold_precisions))
    print("Mean Recall       :", np.mean(fold_recalls))
    print("Mean F1-Score     :", np.mean(fold_f1s))

    print("\n=== Train final model on ALL samples & save ===")
    final_clf = create_model()
    final_clf.fit(X_flat, y)

    model_path = os.path.join(DATASET_DIR, "linearsvc_model.pkl")
    joblib.dump(final_clf, model_path)
    print(f"✅ Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
