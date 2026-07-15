import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page configuration
st.set_page_config(
    page_title="Sentiment AI Chatbot",
    page_icon="🤖"
)

st.title("🤖 Sentiment AI Chatbot")
st.write("Detect Positive, Negative and Neutral sentiments from user messages.")

# User input
user_input = st.text_area("Enter your message")

if st.button("Predict Sentiment"):

    if user_input.strip() == "":
        st.warning("Please enter a message.")

    else:
        text = vectorizer.transform([user_input])
        prediction = model.predict(text)[0]

        st.subheader("Prediction")
        st.write(prediction.capitalize())

        if prediction.lower() == "positive":
            st.success("😊 Positive Sentiment")
            st.write("### Chatbot Response")
            st.write("That's wonderful! 😊 I'm glad you're having a positive experience. Keep smiling!")

        elif prediction.lower() == "negative":
            st.error("😔 Negative Sentiment")
            st.write("### Chatbot Response")
            st.write("I'm sorry you're feeling this way. 💙 I hope things get better soon. Let me know if I can help.")

        elif prediction.lower() == "neutral":
            st.info("😐 Neutral Sentiment")
            st.write("### Chatbot Response")
            st.write("Thank you for sharing your message. How can I assist you further?")

        else:
            st.warning("Unable to determine sentiment.")