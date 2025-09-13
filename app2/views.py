from django.shortcuts import render
from django.http import HttpResponse
from .forms import ColonUploadForm
from .models import UploadedImage
import numpy as np
from PIL import Image as PILImage
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from .gradcam import generate_gradcam_overlay
import os

# Load the pre-trained model
model = load_model('D:\\PDEU\\SEM-2\\NN & DL\\Cancer-Detection-Website\\app2\\best_inception_model.h5')
print('Model Colon Cancer Loaded Successfully')

# Define the target size for image resizing (must match the model input)
target_size = (299, 299)

# Prediction helper
def predict_image(image_arr):
    try:
        prediction = model.predict(image_arr)
        predicted_class = np.argmax(prediction, axis=1)[0]
        classes = ['Benign Cases', 'Malignant Cases']
        probability = float(prediction[0][predicted_class])
        result = classes[predicted_class]
        return result, probability
    except Exception as e:
        print(f'Error during prediction: {str(e)}')
        return f'Error: {str(e)}', 0.0

# Main view for colon cancer detection
def colon_cancer(request):
    if request.method == 'POST':
        form = ColonUploadForm(request.POST, request.FILES)
        if form.is_valid():
            image_file = request.FILES['image']
            image_file.seek(0)

            # Save safely
            new_image = UploadedImage(image=image_file)
            new_image.save()
            img_path = new_image.image.path

            # Preprocess
            img = PILImage.open(img_path).convert("RGB")
            img = img.resize((299, 299))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0) / 255.0

            result, probability = predict_image(img_array)

            new_image.result = result
            new_image.probability = probability
            new_image.save()

            gradcam_path = None
            if result == "Malignant Cases":
                gradcam_result = generate_gradcam_overlay(
                    image_path=img_path,
                    model=model,
                    last_conv_layer_name="mixed10",
                    class_names=["Benign", "Malignant"]
                )
                gradcam_path = gradcam_result[0] if isinstance(gradcam_result, tuple) else gradcam_result

            return render(request, 'result_colon.html', {
                'result': result,
                'probability': round(probability * 100, 2),
                'new_image': new_image,
                'gradcam_path': '/' + gradcam_path if gradcam_path else None
            })
        else:
            return render(request, 'upload_colon.html', {'form': form})
    else:
        form = ColonUploadForm()
    return render(request, 'upload_colon.html', {'form': form})
