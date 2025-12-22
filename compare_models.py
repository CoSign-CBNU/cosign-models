# compare_models.py
# 4개 모델(MLP, RandomForest, LogisticRegression, LinearSVC+Calibration) 성능 비교
# 동일한 5-Fold CV로 평가

import os
import json
import numpy as np
import time

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

# ===== 데이터 경로 =====
DATASET_DIR = "./dataset_out"
X_PATH = os.path.join(DATASET_DIR, "X_seq.npy")
Y_PATH = os.path.join(DATASET_DIR, "y.npy")
LABEL_MAP_PATH = os.path.join(DATASET_DIR, "label_map.json")


# ===== 데이터 로드 =====
def load_data():
    X = np.load(X_PATH)
    y = np.load(Y_PATH)

    with open(LABEL_MAP_PATH, "r", encoding="utf-8") as f:
        lm = json.load(f)
    id2label = {int(k): v for k, v in lm["id2label"].items()}

    print("=== Loaded dataset ===")
    print(f"X shape: {X.shape}  (N, SEQ_LEN, FEAT_DIM)")
    print(f"y shape: {y.shape}  (N,)")
    print(f"Number of classes: {len(id2label)}")
    print(f"Total samples: {len(y)}\n")
    
    return X, y, id2label


# ===== 모델 생성 함수들 =====
def create_mlp():
    """MLP Classifier"""
    return MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        alpha=1e-4,
        learning_rate_init=1e-3,
        max_iter=2000,
        random_state=42,
    )


def create_random_forest():
    """Random Forest Classifier"""
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        max_features="log2",
        min_samples_leaf=2,
        min_samples_split=10,
        n_jobs=-1,
        random_state=42,
    )


def create_logistic():
    """Logistic Regression"""
    return LogisticRegression(
        penalty="elasticnet",
        l1_ratio=0.5,
        solver="saga",
        max_iter=5000,
        C=1.0,
        multi_class="auto",
        random_state=42,
    )


def create_linearsvc_calibrated():
    """LinearSVC + Calibration"""
    base_clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LinearSVC(
            C=0.4,
            class_weight="balanced",
            max_iter=5000,
            random_state=42,
        ),
    )
    return CalibratedClassifierCV(
        estimator=base_clf,
        method="sigmoid",
        cv=3,
    )


# ===== 5-Fold CV 평가 함수 =====
def evaluate_model(model, model_name, X_flat, y):
    """
    주어진 모델을 5-Fold CV로 평가하고 결과 반환
    """
    print(f"\n{'=' * 70}")
    print(f"Evaluating: {model_name}")
    print(f"{'=' * 70}")
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1_score': [],
        'train_time': [],
        'test_time': []
    }
    
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X_flat, y), 1):
        print(f"  Fold {fold_idx}/5...", end=" ", flush=True)
        
        X_train, X_test = X_flat[train_idx], X_flat[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # 학습 시간 측정
        start_train = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_train
        
        # 예측 시간 측정
        start_test = time.time()
        y_pred = model.predict(X_test)
        test_time = time.time() - start_test
        
        # 메트릭 계산
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
        rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
        
        fold_results['accuracy'].append(acc)
        fold_results['precision'].append(prec)
        fold_results['recall'].append(rec)
        fold_results['f1_score'].append(f1)
        fold_results['train_time'].append(train_time)
        fold_results['test_time'].append(test_time)
        
        print(f"Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    
    # 평균 및 표준편차 계산
    summary = {}
    for metric, values in fold_results.items():
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values),
            'values': values
        }
    
    return summary


# ===== 메인 함수 =====
def main():
    # 데이터 로드
    X, y, id2label = load_data()
    N, T, F = X.shape
    X_flat = X.reshape(N, T * F)
    
    # 모델 정의
    models = {
        'MLP': create_mlp(),
        'Random Forest': create_random_forest(),
        'Logistic Regression': create_logistic(),
        'LinearSVC + Calibration': create_linearsvc_calibrated(),
    }
    
    # 각 모델 평가
    all_results = {}
    for model_name, model in models.items():
        results = evaluate_model(model, model_name, X_flat, y)
        all_results[model_name] = results
    
    # ===== 결과 비교 출력 =====
    print("\n\n" + "=" * 100)
    print("MODEL PERFORMANCE COMPARISON (5-Fold Cross-Validation)")
    print("=" * 100)
    
    # 헤더
    print(f"\n{'Model':<25} | {'Accuracy':<20} | {'Precision':<20} | {'Recall':<20} | {'F1-Score':<20}")
    print("-" * 110)
    
    # 각 모델 결과
    for model_name, results in all_results.items():
        acc_mean = results['accuracy']['mean']
        acc_std = results['accuracy']['std']
        prec_mean = results['precision']['mean']
        prec_std = results['precision']['std']
        rec_mean = results['recall']['mean']
        rec_std = results['recall']['std']
        f1_mean = results['f1_score']['mean']
        f1_std = results['f1_score']['std']
        
        print(f"{model_name:<25} | "
              f"{acc_mean:.4f} (±{acc_std:.4f}) | "
              f"{prec_mean:.4f} (±{prec_std:.4f}) | "
              f"{rec_mean:.4f} (±{rec_std:.4f}) | "
              f"{f1_mean:.4f} (±{f1_std:.4f})")
    
    # 학습/추론 시간 비교
    print("\n\n" + "=" * 100)
    print("TRAINING & INFERENCE TIME COMPARISON")
    print("=" * 100)
    print(f"\n{'Model':<25} | {'Train Time (avg)':<20} | {'Test Time (avg)':<20} | {'Total Time':<20}")
    print("-" * 90)
    
    for model_name, results in all_results.items():
        train_mean = results['train_time']['mean']
        test_mean = results['test_time']['mean']
        total_mean = train_mean + test_mean
        
        print(f"{model_name:<25} | "
              f"{train_mean:.4f}s             | "
              f"{test_mean:.4f}s             | "
              f"{total_mean:.4f}s")
    
    # 최고 성능 모델 찾기
    print("\n\n" + "=" * 100)
    print("BEST PERFORMING MODELS")
    print("=" * 100)
    
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        best_model = max(all_results.items(), key=lambda x: x[1][metric]['mean'])
        best_name = best_model[0]
        best_value = best_model[1][metric]['mean']
        best_std = best_model[1][metric]['std']
        
        print(f"Best {metric.capitalize():<12}: {best_name:<25} ({best_value:.4f} ±{best_std:.4f})")
    
    # 상세 Fold별 결과
    print("\n\n" + "=" * 100)
    print("DETAILED FOLD-BY-FOLD RESULTS")
    print("=" * 100)
    
    for model_name, results in all_results.items():
        print(f"\n{model_name}:")
        print(f"  Accuracy per fold:  {[f'{x:.4f}' for x in results['accuracy']['values']]}")
        print(f"  Precision per fold: {[f'{x:.4f}' for x in results['precision']['values']]}")
        print(f"  Recall per fold:    {[f'{x:.4f}' for x in results['recall']['values']]}")
        print(f"  F1-Score per fold:  {[f'{x:.4f}' for x in results['f1_score']['values']]}")


if __name__ == "__main__":
    main()
