import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Load temporal dataset
df = pd.read_csv("lara_temporal_dataset.csv")

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

# 700 features = 10 frames × 70 features
# Use only first 5 frames = 350 features
EARLY_FEATURES = feature_cols[:350]

X = df[EARLY_FEATURES]
y = df["Label"]

print("\n" + "=" * 50)
print("EARLY INTENTION PREDICTION")
print("=" * 50)

print("Observation window: First 5 frames")
print("Features per frame: 70")
print("Total early features:", len(EARLY_FEATURES))

print("\nTotal samples:", len(df))

print("\nClass distribution:")
print(y.value_counts())

# Stratified split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))

# Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced",
    n_jobs=-1
)

print("\nTraining Early Intention Random Forest...")

model.fit(X_train, y_train)

print("Training complete.")

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 50)
print("EARLY INTENTION RESULTS")
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

# Save
joblib.dump(
    model,
    "lara_early_intention_rf.pkl"
)

print("\nModel saved as:")
print("lara_early_intention_rf.pkl")