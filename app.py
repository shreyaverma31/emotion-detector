import tensorflow as tf
import pickle
import numpy as np
import nltk
from nltk.corpus import stopwords
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Load stopwords
nltk.download("stopwords")
stop_words = set(stopwords.words("english"))

# Load trained model
model = tf.keras.models.load_model("emotion_model.keras")

# Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# Emotion label mapping (based on dataset)
emotion_labels = {
    0: "sadness 😢",
    1: "joy 😊",
    2: "love ❤️",
    3: "anger 😠",
    4: "fear 😨",
    5: "surprise 😲"
}

def clean_text(text):
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def predict_emotion(text):
    text = clean_text(text)
    seq = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=100)
    prediction = model.predict(padded)
    label = np.argmax(prediction)
    confidence = np.max(prediction)
    return emotion_labels[label], confidence

# ---------------- MAIN ---------------- #

print("🧠 Emotion Detection App")

print("Type 'exit' to quit\n")

while True:
    user_input = input("Enter a sentence: ")

    if user_input.lower() == "exit":
        print("👋 Bye!")
        break

    emotion, confidence = predict_emotion(user_input)
    print(f"👉 Emotion: {emotion} (confidence: {confidence:.2f})\n")
