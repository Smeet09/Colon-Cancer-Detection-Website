import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Load the pre-trained model
model = load_model('C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\project\\cancer\\app1\\Coloncancer1_model.keras')

# Define the target size for image resizing (must match the input size used during training)
print('5')
target_size = (299, 299)

def preprocess_image(image_path):
    print('6')
    img = image.load_img(image_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize pixel values to between 0 and 1
    return img_array

def predict_image(image):
    try:
        print('7')
        processed_img = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_img)

        # Get the predicted class (0 for Benign, 1 for Malignant)
        predicted_class = np.argmax(prediction, axis=1)[0]

        # Define your classes
        classes = ['Benign Cases', 'Malignant Cases']

        # Display the result based on the predicted class
        return classes[predicted_class], float(prediction[0][predicted_class])

    except Exception as e:
        return 'Error', 0.0
