# Sentiment AI Chatbot

## Project Description
The Sentiment AI Chatbot is a Python-based chatbot that detects the sentiment of user messages using Machine Learning. It classifies text into Positive, Negative, or Neutral sentiments and responds accordingly. The chatbot also stores the chat history and displays a summary when the conversation ends.

## Features
- Detects Positive, Negative, and Neutral sentiments
- Uses Machine Learning (Naive Bayes)
- Uses TF-IDF Vectorizer
- Saves chat history
- Displays conversation summary
- Easy command-line interface

## Technologies Used
- Python
- Scikit-learn
- Pandas
- Joblib
- VS Code

## Project Structure

Sentiment-Analysis-Chatbot/
│
├── chatbot.py
├── train_model.py
├── test_model.py
├── README.md
├── requirements.txt
│
├── dataset/
│   └── sentiment_dataset.csv
│
├── models/
│   ├── sentiment_model.pkl
│   └── vectorizer.pkl
│
├── chat_history/
│   └── chat_log.txt
│
├── notebook/
│   └── sentiment_analysis.ipynb
│
├── images/
│   ├── bar_chart.png
│   ├── pie_chart.png
│   ├── confusion_matrix.png
│   └── wordcloud.png
│
└── report/
    └── Project_Report.pdf

## How to Run

1. Install required packages
pip install -r requirements.txt

2. Train the model
python train_model.py

3. Run the chatbot
python chatbot.py

## Author
Prathwik H Devadiga