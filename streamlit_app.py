import tensorflow as tf
import pickle
import numpy as np
import nltk
import streamlit as st
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

model = tf.keras.models.load_model("emotion_model.keras")

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
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

st.title("🧠 Emotion Detection from Text")
st.write("Enter a sentence and detect emotion")

user_text = st.text_area("Your text here:")

if st.button("Predict Emotion"):
    cleaned = clean_text(user_text)
    seq = tokenizer.texts_to_sequences([cleaned])
    padded = pad_sequences(seq, maxlen=100)
    pred = model.predict(padded)
    label = np.argmax(pred)
    confidence = np.max(pred)

    st.success(f"Emotion: {emotion_labels[label]}")
    st.write(f"Confidence: {confidence:.2f}")
