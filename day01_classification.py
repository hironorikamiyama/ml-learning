
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# 1. データの読み込み
dataset = load_breast_cancer()
X = dataset.data

# 1 = 悪性、0 = 良性 に統一
y = (dataset.target == 0).astype(int)

# 2. 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 3. モデルの構築
# 標準化は学習データだけでfitする（データリーク対策）
model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000),
)

# 4. 学習
model.fit(X_train, y_train)

# 5. 予測
y_pred = model.predict(X_test)

# 6. 評価
print("Accuracy :", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall   :", recall_score(y_test, y_pred))
print("F1       :", f1_score(y_test, y_pred))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=["Benign", "Malignant"],
    )
)


# 判定閾値を変更して比較する
y_proba = model.predict_proba(X_test)[:, 1]

print("\nThreshold Comparison")

for threshold in [0.3, 0.5, 0.7]:
    y_pred_threshold = (y_proba >= threshold).astype(int)

    print(f"\nThreshold: {threshold}")
    print("Precision:", precision_score(y_test, y_pred_threshold))
    print("Recall   :", recall_score(y_test, y_pred_threshold))
    print("F1       :", f1_score(y_test, y_pred_threshold))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred_threshold))
