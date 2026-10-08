
import numpy as np

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# 1. データ取得（DAY 1と同じ陽性の定義）
dataset = load_breast_cancer()
X = dataset.data
y = (dataset.target == 0).astype(int)

# 2. 最初にTest 20%を隔離
X_dev, X_test, y_dev, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. 残り80%をTrain 60% / Validation 20%に分割
X_train, X_val, y_train, y_val = train_test_split(
    X_dev, y_dev,
    test_size=0.25,
    random_state=42,
    stratify=y_dev,
)

print("Train:", len(y_train))
print("Validation:", len(y_val))
print("Test:", len(y_test))

# 4. 学習
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
model.fit(X_train, y_train)

# 5. Validationで閾値を選択
thresholds = [0.3, 0.5, 0.7]
val_proba = model.predict_proba(X_val)[:, 1]

results = []

print("\n=== Validation ===")
for threshold in thresholds:
    pred = (val_proba >= threshold).astype(int)

    precision = precision_score(y_val, pred, zero_division=0)
    recall = recall_score(y_val, pred, zero_division=0)
    f1 = f1_score(y_val, pred, zero_division=0)

    results.append((threshold, f1))

    print(
        f"threshold={threshold:.1f} "
        f"precision={precision:.4f} "
        f"recall={recall:.4f} "
        f"f1={f1:.4f}"
    )

# 今回はValidationのF1最大を採用
# 同点の場合はリスト上の先頭を採用
best_threshold = max(results, key=lambda item: item[1])[0]

print(f"\nSelected threshold: {best_threshold}")

# 6. 選択済み閾値でTestを最終評価
# DEV全体で再学習するため、Validation選択時のモデルとは異なる
final_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)
final_model.fit(X_dev, y_dev)

test_proba = final_model.predict_proba(X_test)[:, 1]
test_pred = (test_proba >= best_threshold).astype(int)

print("\n=== Final Test ===")
print("Accuracy :", accuracy_score(y_test, test_pred))
print("Precision:", precision_score(y_test, test_pred, zero_division=0))
print("Recall   :", recall_score(y_test, test_pred, zero_division=0))
print("F1       :", f1_score(y_test, test_pred, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(y_test, test_pred))

# 7. 5-fold交差検証
# Testデータを含めず、開発用データのみで評価
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

scores = cross_validate(
    make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1000),
    ),
    X_dev,
    y_dev,
    cv=cv,
    scoring={
        "accuracy": "accuracy",
        "f1": "f1",
        "recall": "recall",
    },
)

print("\n=== 5-Fold Cross Validation ===")
for metric in ["test_accuracy", "test_f1", "test_recall"]:
    values = scores[metric]
    print(f"{metric}: {np.round(values, 4)}")
    print(f"  mean={values.mean():.4f}, std={values.std():.4f}")
