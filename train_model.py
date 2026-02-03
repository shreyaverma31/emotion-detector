import pandas as pd
import numpy as np
import nltk
import re
import pickle
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense

print("🚀 Script started...")

# Download stopwords
nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# Load dataset
data = pd.read_csv("emotion.csv")
print("✅ Dataset loaded:", data.shape)

# Clean text
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

data['clean_text'] = data['text'].apply(clean_text)

# Tokenization
tokenizer = Tokenizer(num_words=10000)
tokenizer.fit_on_texts(data['clean_text'])
sequences = tokenizer.texts_to_sequences(data['clean_text'])

X = pad_sequences(sequences, maxlen=100)
y = data['label']

print("✅ Text converted to sequences")
print("Shape of X:", X.shape)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Build model
model = Sequential()
model.add(Embedding(input_dim=10000, output_dim=128, input_length=100))
model.add(LSTM(128))
model.add(Dense(64, activation='relu'))
model.add(Dense(6, activation='softmax'))

model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print("🧠 Starting model training...")

# Train model
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)

# Save model & tokenizer
model.save("emotion_model.keras")

pickle.dump(tokenizer, open("tokenizer.pkl", "wb"))

print("✅ Model training complete and saved!")
