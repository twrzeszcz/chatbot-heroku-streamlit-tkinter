from flask import Flask, render_template, request
import pickle
import numpy as np
import tensorflow as tf
from tensorflow import keras
import json
import random

model = keras.models.load_model('chatbot_model.h5')
intents = json.loads(open('intents.json').read())
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))
lencoder = pickle.load(open('lencoder.pkl', 'rb'))


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

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get', methods=['GET', 'POST'])
def get_bot_response():
    msg = request.args.get('msg')
    return chatbot_response(msg, vectorizer, lencoder, model, intents)

if __name__ == '__main__':
    app.run(debug=True)

