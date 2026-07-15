import streamlit as st

st.set_page_config(page_title="Sentiment AI Chatbot")

st.title("🤖 Sentiment AI Chatbot")
st.write("Welcome to my Sentiment AI Chatbot project.")

user_input = st.text_input("Enter your message:")

if user_input:
    st.write("You entered:", user_input)
    st.success("Chatbot is working successfully!")