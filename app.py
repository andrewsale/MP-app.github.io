from flask import Flask, render_template, url_for, request
from tensorflow.keras.models import load_model
import numpy as np

app = Flask(__name__)

model = load_model('Model-K152')

def speech_predict(speech, model, overlap = 50):
    split_speech = speech.split()
    length = len(split_speech)
    if length < 100:
        segments = [speech]
    else:
        segments = []
        i=0
        while i <= length - 100:
            segments.append(' '.join(split_speech[i:i+100]))
            i+=(100-overlap)
    preds = model.predict(segments, verbose=0)
    pred = np.sum(preds)/len(preds)
    return pred

@app.route("/")
def hello():
    return render_template('home.html', result = None)

@app.post("/predict")
def predict():
    speech = request.form['user-input']
    prediction_prob = speech_predict(speech, model, overlap=50)
    threshold = 0.5

    if prediction_prob < threshold:
        pred = 'Labour'
        prob = 1- prediction_prob
    else:
        pred = 'Conservative'
        prob = prediction_prob

    return render_template('home.html', speech = speech, result = pred, prob=prob)