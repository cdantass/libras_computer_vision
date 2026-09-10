import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("loading dataset...")

df = pd.read_csv("hand_landmarks_dataset.csv")

x = df.drop("label", axis=1)

y = df["label"]

print(f"Total samples: {len(df)}")

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print(f"Training samples: {len(x_train)}")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1)

model.fit(x_train, y_train)
print("Testing model...")

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)
print(f"Model accuracy: {accuracy * 100:.2f}%")

joblib.dump(model, "hand_gesture_model.pkl")

print("Model saved as: hand_gesture_model.pkl")