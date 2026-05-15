

import streamlit as st
import pickle



model = pickle.load(open("sentiment_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))


st.set_page_config(
    page_title="Review Sentiment Analysis",
    layout="centered"
)


st.title("Review sentiment analyzer")

review = st.text_area(
    "Enter review text"
)

if st.button("Predict sentiment"):

        
        review_tfidf = vectorizer.transform([review])

       
        prediction = model.predict(review_tfidf)

       
        if prediction[0] == 0:
            sentiment = "Negative"

        elif prediction[0] == 1:
            sentiment = "Neutral"

        else:
            sentiment = "Positive"

        
        st.success(f"Predicted Sentiment: {sentiment}")