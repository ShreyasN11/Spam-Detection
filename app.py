import streamlit as st
import pickle
from sklearn.feature_extraction.text import CountVectorizer

with open('spam_detector_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('count_vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_email(text):
    text_vectorized = vectorizer.transform([text])
    prediction = model.predict(text_vectorized)
    return 'Spam' if prediction == 1 else 'Not Spam'

st.title('📧 Spam Message Detector')
st.write("Enter the SMS text below to check if it's spam or not.")

email_text = st.text_area("Text")


if st.button('Predict'):
    if email_text:
        result = predict_email(email_text)
        
        if result == 'Spam':
            st.markdown(f"<h2 style='color: red;'>🚨 The email is: <strong>{result}</strong></h2>", unsafe_allow_html=True)
        else:
            st.markdown(f"<h2 style='color: green;'>✅ The email is: <strong>{result}</strong></h2>", unsafe_allow_html=True)
    else:
        st.warning('Please enter some text to analyze.')
