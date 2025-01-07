from flask import Flask, request, jsonify, render_template
import re
from tensorflow.keras.models import load_model
from transformers import RobertaTokenizer, RobertaModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

# Sentiment Prediction
@app.route('/Prediction_sentiment', methods=['POST'])
def predict_sentiment():
    # Labels for sentiment
    sentiment_labels = {0:'neutral' , 1: 'positive', 2: 'negative'}

    # Get JSON data from the request
    data = request.json
    text = data.get('data')  # Extract the value sent from JavaScript

    if not text:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the input text
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+", '', text)  # Remove URLs
        text = re.sub(r"[^a-zA-Z\s!]", '', text)  # Remove special characters
        text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespaces
    else:
        return jsonify({"error": "Invalid input format. Expecting a string."}), 400

    # Tokenize the input sentence
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(text)
    text = tokenizer.texts_to_sequences(text)
    max_length =228


    text = pad_sequences(text,maxlen=max_length)
    # Load the model
    try:
        sentiment_model = load_model('sentiment_model_prototype.keras')
    except Exception as e:
        return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500

    # Predict Sentiment    
    sentiment_probs = sentiment_model.predict(text)
    sentiment_pred = np.argmax(sentiment_probs, axis=1)[0]
    sentiment_label = sentiment_labels[sentiment_pred]
   

    # Return the prediction
    return sentiment_label
@app.route('/Prediction_emotion', methods=['POST'])
def predict_emotion():
    # Labels for emotion
    emotion_labels = {
    0: 'fear',
    1: 'Happiness',
    2: 'anger',
    3: 'joy',   
    4: 'sadness',    
    5: 'neutral', 
}

    # Get JSON data from the request
    data2 = request.json
    text2 = data2.get('data')  # Extract the value sent from JavaScript

    if not text2:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the input text
    if isinstance(text2, str):
        text2 = text2.lower()
        text2 = re.sub(r"http\S+|www\S+", '', text2)  # Remove URLs
        text2 = re.sub(r"[^a-zA-Z\s!]", '', text2)  # Remove special characters
        text2= re.sub(r"\s+", ' ', text2).strip()  # Remove extra whitespaces
    else:
        return jsonify({"error": "Invalid input format. Expecting a string."}), 400

    # Tokenize the input sentence
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(text2)
    text2 = tokenizer.texts_to_sequences(text2)
    max_length =228
    text2 = pad_sequences(text2,maxlen=max_length)
    # Load the model
    
    emotion_model = load_model('emotion_model_prototype.keras')

    # Predict emotion

    emotion_probs = emotion_model.predict(text2)
    emotion_pred = np.argmax(emotion_probs, axis=1)[0]
    emotion_labels = emotion_labels[emotion_pred]

    # Return the prediction
    return emotion_labels
@app.route('/Prediction_Offensive_Language', methods=['POST'])
def predict_Offensive_Language():
    # Labels for hate/offensive language
    hate_mapping = {
    0 :'hate',
    1: 'offensive',
    2:'neither'
    }

    # Get JSON data from the request
    data3 = request.json
    text3 = data3.get('data')  # Extract the value sent from JavaScript

    if not text3:
        return jsonify({"error": "No text provided"}), 400

    # Preprocess the input text
    if isinstance(text3, str):
        text3 = text3.lower()
        text3 = re.sub(r"http\S+|www\S+", '', text3)  # Remove URLs
        text3 = re.sub(r"[^a-zA-Z\s!]", '', text3)  # Remove special characters
        text3= re.sub(r"\s+", ' ', text3).strip()  # Remove extra whitespaces
    else:
        return jsonify({"error": "Invalid input format. Expecting a string."}), 400

    # Tokenize the input sentence
    tokenizer = Tokenizer()

    tokenizer.fit_on_texts(text3)
    text3 = tokenizer.texts_to_sequences(text3)
    max_length =33
    text3 = pad_sequences(text3,maxlen=max_length)
    # Load the model
    
    hate_Language_model = load_model('hate_speech_model_prototype.keras')

    # Predict hate_offensive language

    hate_Language_probs = hate_Language_model.predict(text3)
    hate_Language_pred = np.argmax(hate_Language_probs, axis=1)[0]
    hate_Language_labels =  hate_mapping[hate_Language_pred]

    # Return the prediction
    return hate_Language_labels
if __name__ == '__main__':
    app.run(debug=True, port=5001)