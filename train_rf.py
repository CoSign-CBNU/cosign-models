# train_rf.py
# random forest사용
# AI hub 제공 5개 영상으로 라벨 학습
# 5-fold로 전체 데이터셋 교차검증(tr/ts 분리 X)
import os
import json
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

DATASET_DIR = "./dataset_out"
X_PATH = os.path.join(DATASET_DIR, "X_seq.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")

def load_data():
    # X: (N, SEQ_LEN, FEAT_DIM)
    X = np.load(X_PATH)       # float32일 가능성 큼
    y = np.load(Y_PATH)       # int64

    # label_map 로드 (id2label → 정수 키로 변환)
    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)

    label2id = lm["label2id"]                    # 예: {"1": 0, "2": 1, ...}
    id2label = {int(k): v for k, v in lm["id2label"].items()}  # 예: {0: "1", 1: "2", ...}

    print("=== Loaded dataset ===")
    print(f"X shape: {X.shape}  (N, SEQ_LEN, FEAT_DIM)")
    print(f"y shape: {y.shape}  (N,)")
    print(f"labels : {label2id}")
    return X, y, id2label

def main():
    X, y, id2label = load_data()

    N, T, F = X.shape  # N=샘플 수, T=시퀀스 길이(120), F=피처 차원(192)
    # 랜덤포레스트는 2D 입력(X_samples, n_features)을 기대하므로 flatten
    X_flat = X.reshape(N, T * F)   # (N, 120*192)

    print("\n=== 5-Fold Cross Validation ===")
    print(f"Total samples: {N}")

    # 랜덤포레스트 모델 정의
    clf = RandomForestClassifier(
        n_estimators=300,      # 트리 개수
        max_depth=None,        # 제한 없음 (필요하면 예: 20 정도로 제한 가능)
        n_jobs=-1,             # CPU 전부 사용
        random_state=42,
        class_weight=None      # 라벨 불균형 심하면 "balanced"로 변경 가능
    )

    # 5-Fold 교차검증 (stratified로 라벨 비율 유지)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # 각 fold별로 수동으로 학습 및 평가
    fold_accuracies = []
    fold_precisions = []
    fold_recalls = []
    fold_f1s = []
    
    target_names = [id2label[i] for i in sorted(id2label.keys())]
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), 1):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx}/5")
        print(f"{'='*60}")
        
        # Train/Test split
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 모델 학습
        fold_clf = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            n_jobs=-1,
            random_state=42,
            class_weight=None
        )
        fold_clf.fit(X_train, y_train)
        
        # 예측
        y_pred = fold_clf.predict(X_test)
        
        # 메트릭 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='macro', zero_division=0)
        rec = recall_score(y_test, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
        
        fold_accuracies.append(acc)
        fold_precisions.append(prec)
        fold_recalls.append(rec)
        fold_f1s.append(f1)
        
        print(f"Test samples: {len(y_test)}")
        print(f"Accuracy:  {acc:.4f}")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        
        print(f"\nClassification Report (Fold {fold_idx}):")
        print(classification_report(y_test, y_pred, target_names=target_names, zero_division=0))
        
        print(f"\nConfusion Matrix (Fold {fold_idx}) - 행=실제, 열=예측:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
    
    # 전체 교차검증 결과 요약
    print(f"\n{'='*60}")
    print("=== Cross Validation Summary ===")
    print(f"{'='*60}")
    fold_accuracies = np.array(fold_accuracies)
    fold_precisions = np.array(fold_precisions)
    fold_recalls = np.array(fold_recalls)
    fold_f1s = np.array(fold_f1s)
    
    print(f"Accuracy per fold:  {fold_accuracies}")
    print(f"Mean Accuracy:      {fold_accuracies.mean():.4f} (+/- {fold_accuracies.std():.4f})")
    print(f"Mean Precision:     {fold_precisions.mean():.4f} (+/- {fold_precisions.std():.4f})")
    print(f"Mean Recall:        {fold_recalls.mean():.4f} (+/- {fold_recalls.std():.4f})")
    print(f"Mean F1-Score:      {fold_f1s.mean():.4f} (+/- {fold_f1s.std():.4f})")

    # 전체 데이터로 최종 모델 학습
    print("\n=== Training final model on full dataset ===")
    clf.fit(X_flat, y)
    
    # 전체 데이터에 대한 예측 (참고용)
    y_pred = clf.predict(X_flat)
    acc = accuracy_score(y, y_pred)
    
    print(f"\n{'='*60}")
    print("=== Full Dataset Results (참고용) ===")
    print(f"{'='*60}")
    print(f"Full dataset accuracy (training): {acc * 100:.2f}%")

    print("\nClassification Report on full dataset:")
    print(classification_report(y, y_pred, target_names=target_names, zero_division=0))

    print("\nConfusion Matrix on full dataset (행=실제, 열=예측):")
    print(confusion_matrix(y, y_pred))

    # 원하면 모델 저장 (나중에 inference에서 사용)
    model_path = os.path.join(DATASET_DIR, "rf_model.pkl")
    joblib.dump(clf, model_path)
    print(f"\n✅ Model saved to: {model_path}")

if __name__ == "__main__":
    main()
