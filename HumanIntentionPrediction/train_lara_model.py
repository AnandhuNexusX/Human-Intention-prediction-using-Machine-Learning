import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# =========================================================
# LOAD DATASET
# =========================================================

df = pd.read_csv("lara_temporal_dataset.csv")

print("Total samples:", len(df))
print("Total columns:", len(df.columns))

# =========================================================
# SEPARATE FEATURES AND LABEL
# =========================================================

metadata = [
    "Subject",
    "Recording",
    "StartFrame",
    "Label"
]

feature_cols = [
    c for c in df.columns
    if c not in metadata
]

X = df[feature_cols]
y = df["Label"]

print("Features per sample:", len(feature_cols))

print("\nClass distribution:")
print(y.value_counts())

# =========================================================
# STRATIFIED TRAIN / TEST SPLIT
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# =========================================================
# RANDOM FOREST
# =========================================================

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")

# =========================================================
# PREDICTION
# =========================================================

y_pred = model.predict(X_test)

# =========================================================
# RESULTS
# =========================================================

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("FINAL LARa RANDOM FOREST RESULTS")
print("=" * 50)

print(f"\nAccuracy: {accuracy:.4f}")
print(f"Accuracy percentage: {accuracy * 100:.2f}%")

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        digits=4
    )
)

print("\nConfusion Matrix:")

print(
    confusion_matrix(
        y_test,
        y_pred,
        labels=["Walking", "Standing", "Handling"]
    )
)

# =========================================================
# SAVE MODEL
# =========================================================

joblib.dump(
    model,
    "lara_human_intention_rf.pkl"
)

print("\nModel saved as:")
print("lara_human_intention_rf.pkl")