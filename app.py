import streamlit as st
import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

import pickle
nltk.download('punkt')
nltk.download('stopwords')
nltk.corpus.stopwords.words("english")

#Function to process DF
def transform_text(text):


    text = text.lower()
    text = word_tokenize(text)
    
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    
    text = y.copy()
    y.clear()
    
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y.copy()
    y.clear()
    
    ps = PorterStemmer()

    for i in text:
        y.append(ps.stem(i))
    
    return " ".join(y)


rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection"])

#Home Page
if rad=="Home":
    st.title("Text Analysis App")
    st.image("text-analysis.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")

#Spam Detection Analysis Page
if rad=="Spam or Ham Detection":
    tfidf = pickle.load(open('vectorizer.pkl','rb'))
    model = pickle.load(open('model.pkl','rb'))
    
    st.header("Detect Whether A Text Is Spam Or Ham??")
    input_sms=st.text_area("Enter The Text")
    
    if st.button("Predict"):
        transformed_sent=transform_text(input_sms)
    
        vector_input=tfidf.transform([transformed_sent])
        prediction=model.predict(vector_input)[0]

        if prediction==1:
            st.warning("Spam Text!!")
        elif prediction==0:
            st.success("Ham Text!!")