import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load data and trained model
df = pd.read_csv("lara_temporal_dataset.csv")
model = joblib.load("lara_human_intention_rf.pkl")

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

# Recreate the same 80/20 test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Predict
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("=" * 55)
print("LARa HUMAN INTENTION PREDICTION DEMONSTRATION")
print("=" * 55)

print(f"\nTest accuracy: {accuracy * 100:.2f}%")

print("\nSample predictions:")

for i in range(min(30, len(X_test))):
    print(
        f"Sample {i+1:02d} | "
        f"Actual: {y_test.iloc[i]:10s} | "
        f"Predicted: {y_pred[i]}"
    )

print("\n" + "=" * 55)
print("DEMONSTRATION COMPLETE")
print("=" * 55)