from flask import Flask, render_template, request
import nltk
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random

lemmatizer = WordNetLemmatizer()
model = keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['GET', 'POST'])
def get_bot_response():
    msg = request.args.get('msg')
    return chatbot_response(model, words, msg, intents)

if __name__ == '__main__':
    app.run(debug=True)

