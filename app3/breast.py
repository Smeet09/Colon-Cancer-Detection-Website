#Import_Libraries
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image
import seaborn as sns
import cv2
import random
import os
import imageio
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
from collections import Counter
from skimage.filters import gaussian
from skimage.util import random_noise
import matplotlib.image as mpimg


from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import accuracy_score, recall_score, precision_score, classification_report, confusion_matrix 
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, RepeatedStratifiedKFold
from imblearn.over_sampling import SMOTE


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input, decode_predictions


#data_load
breast_cancer_dir = 'C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\Breast_Cancer_Dataset'

categories = ['Benign Cases', 'Malignant Cases', 'Normal Cases']


#size_of_data
import os
import cv2

def get_image_sizes(directory, categories):
    image_sizes = {category: {} for category in categories}

    for category in categories:
        category_path = os.path.join(directory, category)
        for filename in os.listdir(category_path):
            if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".jpeg"):
                image_path = os.path.join(category_path, filename)

                # Use OpenCV to read the image and get its dimensions
                img = cv2.imread(image_path)
                if img is not None:
                    height, width, _ = img.shape
                    size_key = f"{width} x {height}"

                    if size_key in image_sizes[category]:
                        image_sizes[category][size_key] += 1
                    else:
                        image_sizes[category][size_key] = 1

    return image_sizes

# lung_cancer_sizes = get_image_sizes(lung_cancer_dir, categories)
breast_cancer_sizes = get_image_sizes(breast_cancer_dir, categories)
# colon_cancer_sizes = get_image_sizes(colon_cancer_dir, categories)

# print("Lung Cancer Dataset Image Sizes:", lung_cancer_sizes)
print("Breast Cancer Dataset Image Sizes:", breast_cancer_sizes)
# print("Colon Cancer Dataset Image Sizes:", colon_cancer_sizes)


#data_visualize
import os
import cv2
import matplotlib.pyplot as plt

# lung_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Lung_Cancer_Dataset'
breast_cancer_dir = 'C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\Breast_Cancer_Dataset'
# colon_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Colon_Cancer_Dataset'

# categories_lung = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
categories_breast = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
# categories_colon = ['Benign Cases', 'Malignant Cases']

for category in categories_breast:
    path = os.path.join(breast_cancer_dir, category)
    class_num = categories_breast.index(category)
    for file in os.listdir(path):
        filepath = os.path.join(path, file)
        print(category)
        img = cv2.imread(filepath, 0)
        plt.imshow(img)
        plt.title(category)
        plt.show()
        break
    
    
#data_resize
import os
import cv2
import matplotlib.pyplot as plt

img_size = 299

# lung_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Lung_Cancer_Dataset'
breast_cancer_dir = 'C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\Breast_Cancer_Dataset'
# colon_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Colon_Cancer_Dataset'

# categories_lung = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
categories_breast = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
# categories_colon = ['Benign Cases', 'Malignant Cases']

for categories in [categories_breast]:
    # , categories_lung, categories_colon]:
    for category in categories:
        cnt, samples = 0, 3
        fig, ax = plt.subplots(samples, 3, figsize=(15, 15))
        fig.suptitle(category)

        # if category in os.listdir(lung_cancer_dir):
        #     path = os.path.join(lung_cancer_dir, category)
        if category in os.listdir(breast_cancer_dir):
            path = os.path.join(breast_cancer_dir, category)
        # elif category in os.listdir(colon_cancer_dir):
        #     path = os.path.join(colon_cancer_dir, category)

        class_num = categories.index(category)
        for curr_cnt, file in enumerate(os.listdir(path)):
            filepath = os.path.join(path, file)
            img = cv2.imread(filepath, 0)

            img0 = cv2.resize(img, (img_size, img_size))

            img1 = cv2.GaussianBlur(img0, (5, 5), 0)

            ax[cnt, 0].imshow(img)
            ax[cnt, 0].set_title('Original')

            ax[cnt, 1].imshow(img0)
            ax[cnt, 1].set_title('Resized')

            ax[cnt, 2].imshow(img1)
            ax[cnt, 2].set_title('Blurred')

            cnt += 1
            if cnt == samples:
                break

        plt.show()




#data_train_test_split
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

img_size = 299
num_channels = 3  # Set to 3 if using color images

data = []

# lung_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Lung_Cancer_Dataset'
breast_cancer_dir = 'C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\Breast_Cancer_Dataset'
# colon_cancer_dir = '/Users/rishijoshi/Desktop/College/BREAST_CANCER/DataSet/Colon_Cancer_Dataset'

# categories_lung = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
categories_breast = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
# categories_colon = ['Benign Cases', 'Malignant Cases']

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_size, img_size))
    return img

for categories in [categories_breast]:
    # , categories_breast, categories_colon
    for category in categories:
        # if category in os.listdir(lung_cancer_dir):
            # path = os.path.join(lung_cancer_dir, category)
        if category in os.listdir(breast_cancer_dir):
            path = os.path.join(breast_cancer_dir, category)
        # elif category in os.listdir(colon_cancer_dir):
            # path = os.path.join(colon_cancer_dir, category)

        class_num = categories.index(category)
        for file in os.listdir(path):
            filepath = os.path.join(path, file)
            img = preprocess_image(filepath)
            data.append([img, class_num])

# Shuffle the data
np.random.shuffle(data)

# Separate features and labels
X = np.array([item[0] for item in data])
y = np.array([item[1] for item in data])

# Normalize pixel values
X = X / 255.0

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Check the shape of the data
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#print_train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X, y, random_state=10, stratify=y)

print(len(X_train), X_train.shape)
print(len(X_valid), X_valid.shape)



#Sequential_model_buil
import numpy as np
import tensorflow as tf
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, LearningRateScheduler
from sklearn.metrics import classification_report, confusion_matrix

# Create a Sequential model
model = Sequential()

# Add Convolutional layers
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(299, 299, 3)))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))

# Flatten the output and add Dense layers
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Add dropout with a dropout rate of 0.5
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))  # Add dropout with a dropout rate of 0.5

# Output layer with 3 units (assuming 3 classes) and softmax activation
model.add(Dense(3, activation='softmax'))

# Compile the model
model.compile(optimizer=tf.optimizers.Adam(lr=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Print a summary of the model architecture
model.summary()




#epoch
history = model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_valid, y_valid))




#model_evaluation
y_pred = model.predict(X_valid, verbose=1)
y_pred_bool = np.argmax(y_pred, axis=1)

print(classification_report(y_valid, y_pred_bool))

confusion_mat = confusion_matrix(y_true=y_valid, y_pred=y_pred_bool)
print(confusion_mat)


#model_test_train_val_accuracy
# Assuming you have already trained your model and stored it in the variable 'model'


# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
val_loss, val_accuracy = model.evaluate(X_valid, y_valid)
train_loss, train_accuracy = model.evaluate(X_train, y_train)

# Print the evaluation results
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
print('-' * 20)
print("Validation Loss:", val_loss)
print("Validation Accuracy:", val_accuracy)
print('-' * 20)
print("Training Loss:", train_loss)
print("Trainig Accuracy:", train_accuracy)


#model_save
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten

# Save the model architecture and weights to HDF5 format
model.save('Breastcancer_model.keras')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the pre-trained model
model = load_model('Breastcancer_model.keras')

def preprocess_input(user_input_path, target_size=(299, 299)):
    # Load and preprocess the image
    img = image.load_img(user_input_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values
    return img_array

def predict_breast_cancer(user_input_path, model):
    # Preprocess the input
    img_array = preprocess_input(user_input_path)
    
    # Make a prediction
    predictions = model.predict(img_array)

    # Get the class with the highest probability
    predicted_class = np.argmax(predictions)

    # Display the result based on predicted class
    if predicted_class == 0:
        return "Benign Cases - Non-cancerous"
    elif predicted_class == 1:
        return "Malignant Cases - Cancerous"
    elif predicted_class == 2:
        return "Normal Cases - No evidence of cancer"
    else:
        return "Invalid class prediction"

# Example usage
user_input_path = "C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\CP\\Breast_Cancer_Dataset\\Malignant Cases\\malignant (1)_mask.png"
result = predict_breast_cancer(user_input_path, model)
print(f"The model predicts: {result}")