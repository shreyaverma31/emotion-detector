from fastapi import FastAPI
import tensorflow as tf
import pickle
import numpy as np

app = FastAPI()

model = tf.keras.models.load_model("emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

MAX_LEN = 100
labels = ["Angry", "Fear", "Happy", "Sad", "Surprise"]

@app.get("/")
def root():
    return {"status": "Emotion Detector API running 🚀"}

@app.post("/predict")
def predict(text: str):
    seq = tokenizer.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    return {"emotion": labels[np.argmax(pred)]}
