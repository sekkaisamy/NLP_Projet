import sys
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QVBoxLayout, QWidget,
    QLabel, QLineEdit, QPushButton, QMessageBox
)
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
class PredictionApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Comment classification")
        self.setGeometry(300, 300, 400, 400)
        
        # Set up the layout
        self.central_widget = QWidget()
        self.layout = QVBoxLayout(self.central_widget)      
        # Input label and text box
        self.input_label = QLabel("Emotion & sentiment& Offensive|hate comment classification")
        self.layout.addWidget(self.input_label)
        
        
        self.input_box = QLineEdit("this product is better than the previous one")
        self.layout.addWidget(self.input_box)   
        #Input sentiment               
        self.input_layout_sentiment = QLabel("Type of sentiment")
        self.layout.addWidget(self.input_layout_sentiment) 
        # Output label sentiment
        self.input_label_sentiment = QLineEdit()
        self.layout.addWidget(self.input_label_sentiment)
        
        # Output label Emotion
        self.input_layout_emotion = QLabel("Type of emotion")
        self.layout.addWidget(self.input_layout_emotion )
        self.input_label_emotion = QLineEdit()
        self.layout.addWidget(self.input_label_emotion) 
        # Output label Offensive_Language
        self.input_layout_Offensive_Language = QLabel("Offensive or hate Language")
        self.layout.addWidget(self.input_layout_Offensive_Language)
        self.input_label_Offensive_Language = QLineEdit()
        self.layout.addWidget(self.input_label_Offensive_Language)
        # Predict button
        self.predict_button = QPushButton("Predict")
        self.predict_button.clicked.connect(self.make_prediction_sentiment)
        self.predict_button.clicked.connect(self.make_prediction_emotion)
        self.predict_button.clicked.connect(self.make_prediction_Offensive_Language)
        self.layout.addWidget(self.predict_button)
        # clear all button
        self.Clear_all_button = QPushButton("Clear_all")
        self.Clear_all_button.clicked.connect(self.clear_all)
        self.layout.addWidget(self.Clear_all_button)
        # Set the central widget
        self.setCentralWidget(self.central_widget)
    
    def make_prediction_sentiment(self):
        try:
            # Placeholder for processing the input value
            input_value_sentiment = self.input_box.text()
            text=input_value_sentiment
            # Labels for sentiment
            sentiment_labels = {0:'neutral' , 1: 'positive', 2: 'negative'}
            
            if not text:
                return "error: No text provided"

            # Preprocess the input text
            if isinstance(text, str):
                text = text.lower()
                text = re.sub(r"http\S+|www\S+", '', text)  # Remove URLs
                text = re.sub(r"[^a-zA-Z\s!]", '', text)  # Remove special characters
                text = re.sub(r"\s+", ' ', text).strip()  # Remove extra whitespaces
            else:
                return "error: Invalid input format. Expecting a string."

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
            # Display the result
            self.input_label_sentiment.setText(sentiment_label)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    def make_prediction_emotion(self):
        try:
            # Placeholder for processing the input value
            input_value_emotion = self.input_box.text()
            text2=input_value_emotion
            # Labels for emotion
            emotion_labels = {
            0: 'neutral',
            1 : 'happiness',
            2 :'sadness',
            3 :'anger',
            4 :'surprise',
            5 :'fear',
            6 :'disgust',
            7 :'shame',
            8 :'guilt',
            9 :'joy' 
        }

            
            if not text2:
                return "error: No text provided"

            # Preprocess the input text
            if isinstance(text2, str):
                text2 = text2.lower()
                text2 = re.sub(r"http\S+|www\S+", '', text2)  # Remove URLs
                text2 = re.sub(r"[^a-zA-Z\s!]", '', text2)  # Remove special characters
                text2 = re.sub(r"\s+", ' ', text2).strip()  # Remove extra whitespaces
            else:
                return "error: Invalid input format. Expecting a string."

            # Tokenize the input sentence
            tokenizer = Tokenizer()

            tokenizer.fit_on_texts(text2)
            text2 = tokenizer.texts_to_sequences(text2)
            max_length =228


            text2 = pad_sequences(text2,maxlen=max_length)
            # Load the model
            try:
                emotion_model = load_model('emotion_model_prototype.keras')
            except Exception as e:
                return jsonify({"error": f"Failed to load the model: {str(e)}"}), 500

            # Predict emotion    
            emotion_probs = emotion_model.predict(text2)
            emotion_pred = np.argmax(emotion_probs, axis=1)[0]
            emotion_label = emotion_labels[emotion_pred]
            # Display the result
            self.input_label_emotion.setText(emotion_label)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid number.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    def make_prediction_Offensive_Language(self):
        try:
            # Placeholder for processing the input value
            input_value_Offensive_Language = self.input_box.text()
            text3=input_value_Offensive_Language
            # Labels for hate/Offensive_Language 
            hate_mapping = {
                0 :'hate',
                1 :'offensive',
                2 : 'neither'
            }

            
            if not text3:
                return "error: No text provided"

            # Preprocess the input text
            if isinstance(text3, str):
                text3 = text3.lower()
                text3 = re.sub(r"http\S+|www\S+", '', text3)  # Remove URLs
                text3 = re.sub(r"[^a-zA-Z\s!]", '', text3)  # Remove special characters
                text3 = re.sub(r"\s+", ' ', text3).strip()  # Remove extra whitespaces
            else:
                return "error: Invalid input format. Expecting a string."

            # Tokenize the input sentence
            tokenizer = Tokenizer()

            tokenizer.fit_on_texts(text3)
            text3 = tokenizer.texts_to_sequences(text3)
            max_length =33


            text3 = pad_sequences(text3,maxlen=max_length)
            # Load the model
            hate_model = load_model('hate_speech_model_prototype.keras')
            # Predict hate/Offensive_Language  
            hate_probs = hate_model.predict(text3)
            hate_pred = np.argmax(hate_probs, axis=1)[0]
            hate_label = hate_mapping[hate_pred]
            # Display the result
            self.input_label_Offensive_Language.setText(hate_label)
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "Please enter a valid text.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An unexpected error occurred: {str(e)}")
    def clear_all(self) :
        self.input_box .setText("")  
        self.input_label_sentiment.setText("")
        self.input_label_emotion.setText("")
        self.input_label_Offensive_Language.setText("")
      
 
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictionApp()
    window.show()
    sys.exit(app.exec())
