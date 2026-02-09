import streamlit as st
import numpy as np
import pickle
import os
import nltk
import tensorflow as tf

from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.data.path.append(os.path.join(os.getcwd(), "nltk_data"))

MAX_LEN = 100
labels = ["Angry 😠", "Fear 😨", "Happy 😊", "Sad 😢", "Surprise 😲"]

st.set_page_config(page_title="Emotion Detector", page_icon="🧠")
st.title("🧠 Emotion Detection from Text")

@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.keras")

stop_words = load_stopwords()
model = load_emotion_model()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

def clean_text(text):
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)

text = st.text_area("Enter text here:")

if st.button("Predict Emotion"):
    if not text.strip():
        st.warning("⚠️ Please enter some text")
    else:
        seq = tokenizer.texts_to_sequences([clean_text(text)])
        padded = pad_sequences(seq, maxlen=MAX_LEN)
        pred = model.predict(padded)
        emotion = labels[np.argmax(pred)]
        st.success(f"🎯 Predicted Emotion: **{emotion}**")
