# 🐦 Tweets Sentiment Analyzer

A Streamlit-based web app that performs sentiment analysis on tweets using Natural Language Processing (NLP) techniques and Logistic Regression. The app allows users to input custom tweets and instantly receive sentiment predictions (positive, negative, or neutral).

## 🔍 Features
- Clean and preprocess tweets
- TF-IDF vectorization of tweet text
- Sentiment prediction using Logistic Regression
- Display model evaluation (accuracy, F1 score, confusion matrix)
- Interactive Streamlit UI

## 📦 Tech Stack
- Python
- Streamlit
- Scikit-learn
- Pandas
- NLTK

## 🚀 Setup Instructions

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/tweets-sentiment-analyzer.git
   cd tweets-sentiment-analyzer
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a `Tweets.csv` file in the same directory with columns like `text` and `airline_sentiment`.

4. Run the app:
   ```bash
   streamlit run main.py
   ```

## 📝 License
MIT License
