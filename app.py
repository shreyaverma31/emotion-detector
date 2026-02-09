from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import nltk

from nltk.corpus import stopwords
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences

nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

MAX_LEN = 100
labels = ["Angry 😠", "Fear 😨", "Happy 😊", "Sad 😢", "Surprise 😲"]

app = FastAPI(title="Emotion Detection API")

# Load model once
model = load_model("emotion_model.keras")

with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

class TextInput(BaseModel):
    text: str

def clean_text(text):
    text = text.lower()
    return " ".join(w for w in text.split() if w not in stop_words)

@app.get("/")
def home():
    return {"message": "Emotion Detector API is running 🚀"}

@app.post("/predict")
def predict_emotion(data: TextInput):
    seq = tokenizer.texts_to_sequences([clean_text(data.text)])
    padded = pad_sequences(seq, maxlen=MAX_LEN)
    pred = model.predict(padded)
    emotion = labels[int(np.argmax(pred))]
    confidence = float(np.max(pred))

    return {
        "emotion": emotion,
        "confidence": round(confidence, 2)
    }
