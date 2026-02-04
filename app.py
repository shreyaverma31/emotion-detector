import pickle
import numpy as np
import nltk
import streamlit as st

from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

# Download stopwords once
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

@st.cache_resource
def load_emotion_model():
    return load_model("emotion_model.keras")

model = load_emotion_model()

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

emotion_labels = {
    0: "Sadness 😢",
    1: "Joy 😊",
    2: "Love ❤️",
    3: "Anger 😠",
    4: "Fear 😨",
    5: "Surprise 😲"
}

def clean_text(text):
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)

st.title("🧠 Emotion Detection from Text")
user_text = st.text_area("Enter your sentence")

if st.button("Predict Emotion"):
    if user_text.strip():
        seq = tokenizer.texts_to_sequences([clean_text(user_text)])
        padded = pad_sequences(seq, maxlen=100)
        pred = model.predict(padded)
        st.success(f"Emotion: {emotion_labels[np.argmax(pred)]}")
