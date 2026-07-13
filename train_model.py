import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Create models folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Load dataset
data = pd.read_csv("dataset/sentiment_dataset.csv")

# Check required columns
if "text" not in data.columns or "sentiment" not in data.columns:
    raise ValueError("Dataset must contain 'text' and 'sentiment' columns.")

# Features and labels
X = data["text"]
y = data["sentiment"]

# Create vectorizer
vectorizer = TfidfVectorizer()

# Transform text
X = vectorizer.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = MultinomialNB()
model.fit(X_train, y_train)

# Test accuracy
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Save files
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("===================================")
print("Training completed successfully!")
print(f"Accuracy: {accuracy*100:.2f}%")
print("Saved: models/sentiment_model.pkl")
print("Saved: models/vectorizer.pkl")
print("===================================")