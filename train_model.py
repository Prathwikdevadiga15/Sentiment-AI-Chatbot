import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Create models folder
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/sentiment_dataset.csv")

# Check required columns
if "text" not in data.columns or "sentiment" not in data.columns:
    raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")

# Remove empty rows
data = data.dropna()

# Features and labels
X = data["text"].astype(str)
y = data["sentiment"].astype(str)

# Split dataset (balanced)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# TF-IDF Vectorizer
vectorizer = TfidfVectorizer(lowercase=True, stop_words="english")

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("=" * 40)
print("Training completed successfully!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("Saved: models/sentiment_model.pkl")
print("Saved: models/vectorizer.pkl")
print("=" * 40)