import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import zipfile
import os

zip_path = "neural_network/cats_dogs_snakes.zip"
extract_path = "animals_data"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_path)


data_folder = os.path.join(extract_path, "dataset")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2, 
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = train_datagen.flow_from_directory(
    data_folder,
    target_size=(256,256),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

num_classes = len(train_generator.class_indices)

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
    MaxPooling2D(2,2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(train_generator, validation_data=val_generator, epochs=20)


model.save("neural_network/animal_classifier_cnn.h5")


from tensorflow.keras.preprocessing import image
import numpy as np

def predict_image(path):
    img = image.load_img(path, target_size=(256,256))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    class_name = [name for name, idx in train_generator.class_indices.items() if idx==class_idx][0]
    print(f"Prediction: {class_name}")

while True:
    img_path = input("Enter path to an image to classify (or 'exit' to quit): ")
    if img_path.lower() == "exit":
        break
    if os.path.exists(img_path):
        predict_image(img_path)
    else:
        print("File not found. Try again.")
