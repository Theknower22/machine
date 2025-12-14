import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------
# 1) Load dataset
# -----------------------
DATASET_PATH = "dataset.csv"
df = pd.read_csv(DATASET_PATH)

df["code"] = df["code"].fillna("").astype(str)
df = df[df["code"].str.strip().str.len() > 0]
df["label"] = df["label"].astype(int)

# -----------------------
# 2) Split data
# -----------------------
X = df["code"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -----------------------
# 3) Vectorizer
# -----------------------
vectorizer = TfidfVectorizer(max_features=50000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -----------------------
# 4) Train model
# -----------------------
model = LogisticRegression(max_iter=2000)
model.fit(X_train_vec, y_train)

# -----------------------
# 5) Evaluation
# -----------------------
preds = model.predict(X_test_vec)
print(classification_report(y_test, preds))

# -----------------------
# 6) Save model + vectorizer
# -----------------------
os.makedirs("models", exist_ok=True)

joblib.dump(model, "models/vuln_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("Model and vectorizer saved successfully.")
