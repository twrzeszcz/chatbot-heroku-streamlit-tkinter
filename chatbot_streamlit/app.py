import nltk
nltk.download('wordnet')
nltk.download('punkt')
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import keras
import streamlit as st
import json
import random

lemmatizer = WordNetLemmatizer()

def load_files():
    model = keras.models.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    words = pickle.load(open('words.pkl', 'rb'))
    classes = pickle.load(open('classes.pkl', 'rb'))

    return model, intents, words, classes

model, intents, words, classes = load_files()

def preprocess(sentence, words):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    bag = [1 if word in sentence_words else 0 for word in words]
    return bag

def predict_class(sentence, words, model):
    prep = preprocess(sentence, words)
    yhat = model.predict([prep])
    pred = {'tag': classes[np.argmax(yhat)], 'prob': np.round(np.max(yhat), 2)}
    return pred

def chatbot_response(model, words, sentence, intents):
    pred = predict_class(sentence, words, model)
    for i in intents['intents']:
        if i['tag'] == pred['tag']:
            result = random.choice(i['responses'])
            break
    return result


st.title('Simple Chatbot')

if st.checkbox('Open chat'):
    msg = st.text_input('You: ')
    if st.button('Send'):
        response = chatbot_response(model, words, msg, intents)
        st.text_input('Chatbot: ', value=response)
