import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

model = load_model("neural_network/animal_classifier_cnn.h5")
class_names = ['cats', 'dogs', 'snakes']

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(256,256))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    print(f"Prediction: {class_names[class_idx]}")

while True:
    img_path = input("Enter path to an image to classify (or 'exit' to quit): ")
    if img_path.lower() == "exit":
        break
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print("File not found. Try again.")
