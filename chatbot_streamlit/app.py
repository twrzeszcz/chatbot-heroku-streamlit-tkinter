import pickle
import numpy as np
import keras
import streamlit as st
import json
import random


def load_files():
    model = keras.models.load_model('chatbot_model.h5')
    intents = json.loads(open('intents.json').read())
    vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
    lencoder = pickle.load(open('lencoder.pkl', 'rb'))

    return model, intents, vectorizer, lencoder

model, intents, vectorizer, lencoder = load_files()


def predict_class(sentence, vectorizer, lencoder, model):
    yhat = model.predict(vectorizer.transform([sentence]).todense())
    tag = lencoder.inverse_transform([np.argmax(yhat)])[0]
    pred = {'tag': tag, 'prob': np.round(np.max(yhat), 2)}
    return pred

def chatbot_response(sentence, vectorizer, lencoder, model, intents):
    pred = predict_class(sentence, vectorizer, lencoder, model)
    for i in intents['intents']:
        if i['tag'] == pred['tag']:
            result = random.choice(i['responses'])
            break
    return result


st.title('Simple Chatbot')

if st.checkbox('Open chat'):
    msg = st.text_input('You: ')
    if st.button('Send'):
        response = chatbot_response(msg, vectorizer, lencoder, model, intents)
        st.text_input('Chatbot: ', value=response)
