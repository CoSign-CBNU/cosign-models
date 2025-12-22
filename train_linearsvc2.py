# train_linearsvc_with_search.py

import os
import json
import numpy as np
import joblib

from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    StratifiedKFold,
    RandomizedSearchCV
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from scipy.stats import uniform

# ===== 경로 설정 =====
DATASET_DIR = "./dataset_out"
X_PATH = os.path.join(DATASET_DIR, "X_seq.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")


# ===== 데이터 로드 =====
def load_data():
    X = np.load(X_PATH)  # (N, T, F)
    y = np.load(Y_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    print("=== Loaded dataset ===")
    print(f"X shape: {X.shape} (N, SEQ_LEN, FEAT_DIM)")
    print(f"y shape: {y.shape} (N,)")
    print(f"labels: {lm['label2id']}")
    return X, y, id2label


# ===== 기본 모델 생성 =====
def create_pipeline(C=1.0):
    return make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearSVC(
            C=C,
            class_weight="balanced",
            max_iter=8000,
            random_state=42,
        )
    )


def main():
    X, y, id2label = load_data()
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)

    print("\n=== Hyperparameter Search (LinearSVC) ===")

    # --- 1) RandomizedSearchCV 설정 ---
    base_pipeline = create_pipeline()

    param_dist = {
        "linearsvc__C": uniform(0.1, 1.0),
        "linearsvc__loss": ["hinge", "squared_hinge"],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    search = RandomizedSearchCV(
        estimator=base_pipeline,
        param_distributions=param_dist,
        n_iter=10,
        scoring="f1_macro",   # 15클래스 → macro F1이 제일 안정적
        n_jobs=-1,
        cv=cv,
        verbose=1,
        random_state=42,
    )

    search.fit(X_flat, y)

    print("\n=== Best Hyperparameters ===")
    print(search.best_params_)
    print(f"Best CV Macro-F1: {search.best_score_:.4f}")

    best_C = search.best_params_["linearsvc__C"]

    # --- 2) Best C로 다시 파이프라인 구성 ---
    final_pipeline = create_pipeline(C=best_C)

    # --- 3) 전체 데이터로 학습 ---
    print("\n=== Train final model on ALL samples ===")
    final_pipeline.fit(X_flat, y)

    # --- 4) 저장 ---
    model_path = os.path.join(DATASET_DIR, "linearsvc_best_model.pkl")
    joblib.dump(final_pipeline, model_path)

    print(f"\n✅ Final model saved to: {model_path}")


if __name__ == "__main__":
    main()
