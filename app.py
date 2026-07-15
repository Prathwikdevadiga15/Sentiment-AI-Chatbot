import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page title
st.set_page_config(page_title="Sentiment AI Chatbot", page_icon="🤖")

st.title("🤖 Sentiment AI Chatbot")
st.write("Enter a message below and click Predict to detect its sentiment.")

# User input
user_input = st.text_area("Enter your message")

if st.button("Predict Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter a message.")
    else:
        text = vectorizer.transform([user_input])
        prediction = model.predict(text)[0]
        st.write("Prediction:",prediction)

        if prediction.lower() == "positive":
            st.success("😊 Positive Sentiment")
            st.write("Response: Awesome! I'm happy for you.")

        elif prediction.lower() == "negative":
            st.error("😔 Negative Sentiment")
            st.write("Response: I'm here to listen if you want to share more.")

        else:
            st.info("😐 Neutral Sentiment")
            st.write("Response: Thanks for sharing.")