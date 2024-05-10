from django.shortcuts import render

# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render
from .forms import ColonUploadForm
from .models import UploadedImage
import numpy as np
from PIL import Image
from keras.preprocessing import image
from django.conf import settings



from .forms import ColonUploadForm
from .models import UploadedImage
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from django.shortcuts import render
from PIL import Image as PILImage

# Load the pre-trained model
model = load_model('C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\project\\cancer2\\app2\\Coloncancer11_model.keras')
print('Model Colon Cancer Successful')
# Define the target size for image resizing (must match the input size used during training)
target_size = (299, 299)

# Define the predict_image function
def predict_image(image_arr):
    try:
        processed_img = image_arr

        if processed_img is not None:
            # Make prediction
            prediction = model.predict(processed_img)

            # Get the predicted class (0 for Benign, 1 for Malignant)
            predicted_class = np.argmax(prediction, axis=1)[0]

            # Define your classes
            classes = ['Benign Cases', 'Malignant Cases']

            # Return the result and probability
            probability = float(prediction[0][predicted_class])
            result = classes[predicted_class]

            return result, probability
        else:
            print("Error: Image preprocessing failed")
            return 'Error: Image preprocessing failed', 0.0
    except Exception as e:
        print(f'Error during prediction: {str(e)}')
        return f'Error during prediction: {str(e)}', 0.0

# Define the colon_cancer view
# Define the colon_cancer view
def colon_cancer(request):
    if request.method == 'POST':
        form = ColonUploadForm(request.POST, request.FILES)
        if form.is_valid():
            # Get the uploaded image file from the form
            image_file = request.FILES.get('image')

            if image_file:
                # Create a new UploadedImage instance
                new_image = UploadedImage(image=image_file)

                # Save the uploaded image instance
                new_image.save()

                # Load the uploaded image
                img = PILImage.open(new_image.image)

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

                # Call predict_image and get the result and probability
                result, probability = predict_image(image_arr=img_array)

                # Update the UploadedImage instance with result and probability
                new_image.result = result
                new_image.probability = probability

                # Save the Updated UploadedImage instance
                new_image.save()

                # Pass the result and probability to the template context
                return render(request, 'result_colon.html', {'result': result, 'probability': probability, 'new_image':new_image})
            else:
                return HttpResponse('Error: Image file not found')
    else:
        form = ColonUploadForm()
    return render(request, 'upload_colon.html', {'form': form})
