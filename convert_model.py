import tensorflow as tf

model = tf.keras.models.load_model("emotion_model_tf")
model.save("emotion_model.keras")
print("✅ Converted and saved as emotion_model.keras")
