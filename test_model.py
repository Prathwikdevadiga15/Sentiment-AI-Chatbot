import joblib

model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

print("Model loaded successfully!")
print("Vectorizer loaded successfully!")