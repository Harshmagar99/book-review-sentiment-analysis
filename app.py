import streamlit as st
from transformers import pipeline

classifier = pipeline(
    "sentiment-analysis",
    model = 'saved_model',
    tokenizer = 'saved_model'
)


st.title("Book review sentiment analysis")

review = st.text_area("Enter the review")

if st.button("Predict"):
    result = classifier(review)

    label = result[0]['label']
    score = result[0]['score']

    if label == "LABEL_0":
        st.error("Negative")
    
    elif label == "LABEL_1":
        st.warning("Neutral")

    else:
        st.success("Positive")

    st.metric(label="Model Confidence", value=f"{score*100:.2f}%")

