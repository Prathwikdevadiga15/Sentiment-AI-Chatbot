import joblib
from datetime import datetime
import os

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Create chat history folder
os.makedirs("chat_history", exist_ok=True)
chat_file = "chat_history/chat_log.txt"

positive_count = 0
negative_count = 0
neutral_count = 0

print("=" * 50)
print("      Welcome to Sentiment AI Chatbot")
print("=" * 50)
print("Type 'exit' anytime to close the chatbot.\n")

while True:

    # Get user input
    message = input("You: ")

    # Exit condition
    if message.lower() == "exit":
        print("\nBot: Thank you for chatting with me. Goodbye!")
        break

    # Convert text to vector
    text_vector = vectorizer.transform([message])

    # Predict sentiment
    prediction = model.predict(text_vector)[0]

    # Generate response
    if prediction.lower() == "positive":
        sentiment = "Positive 😊"
        mood = "😊 Happy"
        response = "Awesome! I'm happy for you."
        positive_count += 1

    elif prediction.lower() == "negative":
        sentiment = "Negative 😔"
        mood = "😔 Sad"
        response = "I'm sorry you're feeling this way. I'm here to listen."
        negative_count += 1

    else:
        sentiment = "Neutral 😐"
        mood = "😐 Calm"
        response = "Thanks for sharing."
        neutral_count += 1

    # Display result
    print("\nSentiment Detected :", sentiment)
    print("Mood               :", mood)
    print("Bot                :", response)
    print()

    # Save chat history
    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")

    with open(chat_file, "a", encoding="utf-8") as file:
        file.write(f"Time: {current_time}\n")
        file.write(f"User: {message}\n")
        file.write(f"Sentiment: {sentiment}\n")
        file.write(f"Mood: {mood}\n")
        file.write(f"Bot: {response}\n")
        file.write("-" * 50 + "\n")

# Chat summary
total = positive_count + negative_count + neutral_count