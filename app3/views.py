from django.shortcuts import render
from django.http import JsonResponse
from django.conf import settings
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image as PILImage
import os

from .forms import BreastCancerImageUploadForm
from .models import PredictedBreastCancer

model = load_model('Breastcancer_model.keras')
print('Breast cancer model loaded')

target_size = (299, 299)

def predict_breast_cancer(img_array):
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

def display_result(request, result, predicted_image):
    return render(request, 'result_breast.html', {'result': result, 'predicted_image': predicted_image})

def predict(request):
    if request.method == 'POST' and request.FILES.get('image'):
        image_file = request.FILES['image']
        img = PILImage.open(image_file)

        # Center crop the image
        width, height = img.size
        left = (width - min(width, height)) / 2
        top = (height - min(width, height)) / 2
        right = (width + min(width, height)) / 2
        bottom = (height + min(width, height)) / 2
        img = img.crop((left, top, right, bottom))

        # Resize the image to the target size
        img = img.resize(target_size)

        # Convert the image to an array
        img_array = image.img_to_array(img)

        # Expand the dimensions to match the input shape expected by the model
        img_array = np.expand_dims(img_array, axis=0)

        # Normalize pixel values to between 0 and 1
        img_array /= 255.0

        # Get prediction result
        result = predict_breast_cancer(img_array)
        
        # Save prediction result to the database
        predicted_image = PredictedBreastCancer(image=image_file, result=result)
        predicted_image.save()

        # Display the result
        # return render(request,  result, predicted_image)
        return render(request, 'result_breast.html', {'result': result, 'predicted_image': predicted_image})
    else:
        form = BreastCancerImageUploadForm()
        return render(request, 'breast_cancer.html', {'form': form})
