import streamlit as st
import tensorflow as tf
import numpy as np
import pickle

# Load model
model = tf.keras.models.load_model("emotion_model_tf")  # or .keras if working

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100
labels = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

st.set_page_config(page_title="Emotion Detector", page_icon="🧠")

st.title("🧠 Emotion Detection from Text")

text = st.text_area("Enter text here:")

if st.button("Predict Emotion"):
    if text.strip() == "":
        st.warning("⚠️ Please enter some text")
    else:
        # Preprocess text
        seq = tokenizer.texts_to_sequences([text])
        padded = tf.keras.preprocessing.sequence.pad_sequences(
            seq, maxlen=MAX_LEN
        )

        # Predict
        pred = model.predict(padded)
        emotion = labels[np.argmax(pred)]

        st.success(f"🎯 Predicted Emotion: **{emotion}**")
