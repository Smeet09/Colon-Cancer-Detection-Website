# import libraries
import os                       # for working with files
import numpy as np              # for numerical computations
import pandas as pd             # for working with dataframes
import seaborn as sns
import tensorflow as tf         # TensorFlow module 
import matplotlib.pyplot as plt # for plotting information on graphs and images using tensors
from tensorflow import keras
from PIL import Image           # for checking images
import itertools
from sklearn.metrics import classification_report, confusion_matrix

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Define colon cancer dataset directory
colon_cancer_dir = 'C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\colon_image_sets'

# Define categories
categories = ['colon_n', 'colon_aca']

# Function to show image
def show_image(image, label_batch):
    labels = train.class_indices
    for label in label_batch:
        try:
            label = np.argmax(label)
            print("Label: " + labels[label])
        except KeyError:
            print("Label index not found:", label)
        plt.imshow(image[0])
        plt.show()

# Data visualization
train = keras.preprocessing.image.ImageDataGenerator(rescale=1./255).flow_from_directory(
    colon_cancer_dir,
    target_size=(299, 299),
    batch_size=1,
    shuffle=False
)

for i in range(6):
    image_batch, label_batch = next(train)
    show_image(image_batch, label_batch)

# Data modeling
train_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    validation_split=0.25
)

valid_gen = keras.preprocessing.image.ImageDataGenerator(
    rescale=1./255,
    validation_split=0.25
)

train_data = train_gen.flow_from_directory(
    colon_cancer_dir,
    subset='training',
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=True
)

val_data = valid_gen.flow_from_directory(
    colon_cancer_dir,
    subset='validation',
    target_size=(299, 299),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Sequential model
model_1 = keras.models.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(299, 299, 3)),
    keras.layers.Dropout(0.1),
    keras.layers.MaxPooling2D(),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.Dropout(0.2),
    keras.layers.MaxPooling2D(),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(2, activation='softmax')
])

model_1.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

model_1.summary()

# Training
history = model_1.fit_generator(
    train_data,
    validation_data=val_data,
    epochs=10
)

# Model evaluation
y_pred = model_1.predict(val_data, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)
y_true = val_data.classes

print(classification_report(y_true, y_pred_bool))
confusion_mat = confusion_matrix(y_true, y_pred_bool)
print(confusion_mat)

# Train-test accuracy
test_loss, test_accuracy = model_1.evaluate(val_data, verbose=1)
train_loss, train_accuracy = model_1.evaluate(train_data, verbose=1)
val_loss, val_accuracy = model_1.evaluate(val_data, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print('-' * 20)
print("Training Loss:", train_loss)
print("Training Accuracy:", train_accuracy)
print('-' * 20)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)

# Save the model
model_1.save('Coloncancer11_model.keras')

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('Coloncancer11_model.keras')

# Define the target size for image resizing (must match the input size used during training)
target_size = (299, 299)

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array

def predict_image(image_path):
    try:
        processed_img = preprocess_image(image_path)

        # Make prediction
        prediction = model.predict(processed_img)

        # Get the predicted class (0 for Benign, 1 for Malignant)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Display the result based on the predicted class
        if predicted_class == 0:
            print("You have non-spreadable cancer, so you belong to the 'Benign Cases' category.")
            print("Benign Cases: Benign tumors are not cancerous and usually do not spread to surrounding tissues.")
        elif predicted_class == 1:
            print("You have spreadable cancer, so you belong to the 'Malignant Cases' category.")
            print("Malignant Cases: Malignant tumors are cancerous and can invade nearby tissues and spread to other parts of the body.")
        
        print("Probability:", float(prediction[0][predicted_class]))

    except Exception as e:
        print("Error:", str(e))

if __name__ == "__main__":
    # Prompt user to enter the path of an image file
    image_path = "C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\colon_image_sets\\colon_aca\\colonca3.jpeg"
    
    # Make predictions
    predict_image(image_path)