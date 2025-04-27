import streamlit as st
import pickle
import nltk
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def transform_text(text):

    # lower-case
    text = text.lower()

    # tokenization
    text = nltk.word_tokenize(text)

    # removing special charchters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    # removing stopwords and punctuations
    for i in text:
        if i not in stopwords.words("english") and i not in string.punctuation:
            y.append(i)
    
    text = y[:]
    y.clear()
    
    # stemming
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    
    return " ".join(y)

tfidf = pickle.load(open("Vectorizer.pkl","rb"))
model = pickle.load(open("model.pkl","rb"))

st.title("E-mail Spam Classifier")

input_email = st.text_area("Enter the message")

if st.button("Predict"):

    # 1. preprocess
    transformed_email = transform_text(input_email)
    # 2. Vectorize
    vector_input = tfidf.transform([transformed_email])
    # 3. predict
    result = model.predict(vector_input)[0]
    # 4. Display
    if result == 1:
        st.header("Spam")
    else:
        st.header("Not Spam")