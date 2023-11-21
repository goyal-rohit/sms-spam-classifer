import streamlit as st
#import numpy as np
#import pandas as pd
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from sklearn.feature_extraction.text import TfidfVectorizer
#from sklearn.model_selection import train_test_split
#from sklearn.preprocessing import LabelEncoder
#from sklearn.naive_bayes import MultinomialNB
import pickle


#Function to process DF
def transform_text(text):


    text = text.lower()
    text = nltk.word_tokenize(text)
    
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

# #Importing csv File
# df = pd.read_csv('spam.csv',encoding='Windows-1252')

# #Data Cleaning
# df.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 4'],inplace=True)
# df.rename(columns={'v1':'target','v2':'text'},inplace=True)
# encoder = LabelEncoder()
# df['target'] = encoder.fit_transform(df['target'])
# df = df.drop_duplicates(keep='first')
# df['transformed_text'] = df['text'].apply(transform_text)

# #Model Building
# tfidf = TfidfVectorizer(max_features=3000)
# X= tfidf.fit_transform(df['transformed_text']).toarray()
# y = df['target']
# X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=2)
# mnb = MultinomialNB()
# mnb.fit(X_train,y_train)


# rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection","Sentiment Analysis","Stress Detection","Hate and Offensive Content Detection","Sarcasm Detection"])
rad=st.sidebar.radio("Navigation",["Home","Spam or Ham Detection"])

#Home Page
if rad=="Home":
    st.title("Text Analysis App")
    st.image("text-analysis.jpg")
    st.text(" ")
    st.text("The Following Text Analysis Options Are Available->")
    st.text(" ")
    st.text("1. Spam or Ham Detection")
    # st.text("2. Sentiment Analysis")
    # st.text("3. Stress Detection")
    # st.text("4. Hate and Offensive Content Detection")
    # st.text("5. Sarcasm Detection")


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