import pandas as pd
import streamlit as st
import nltk 
import re
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')

df = pd.read_csv('Tweets.csv')

stop_words = set(stopwords.words('english'))             #Instead of downloading stopwords every time, we load them once


def clean_text(text):
    text = text.lower()                                
    text = re.sub(r"http\S+|www\S+", "", text)          
    text = re.sub(r"@\w+|#\w+", "", text)                    
    text = re.sub(r"[^a-zA-Z\s]", "", text)
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

df['cleaned_text'] = df['text'].apply(clean_text)

vectorize = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
x = vectorize.fit_transform(df['cleaned_text'])
y = df['airline_sentiment']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=1000)
model.fit(x_train,y_train)
y_pred = model.predict(x_test)

st.title("Tweets Sentiment Analysis")
user_input = st.text_area("Enter a tweet to analyze its sentiment:")
if st.button("Analyze Tweet"):
    if user_input.strip() == "":
        st.error("Please enter a tweet.")
    else:
     with st.spinner("Analyzing..."):
         user_input = clean_text(user_input)
         vectorized_input = vectorize.transform([user_input])
         prediction = model.predict(vectorized_input)
         st.success(f"Predicted Sentiment: {prediction[0]}")

evaluate = st.checkbox("Evaluate Model")
if evaluate:
    st.subheader("Model Evaluation")
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"F1 Score: {f1:.2f}")
    st.write("Confusion Matrix:")
    st.write(cm)

if st.checkbox("Show sample data"):
    st.dataframe(df[['text', 'airline_sentiment']].head())
    