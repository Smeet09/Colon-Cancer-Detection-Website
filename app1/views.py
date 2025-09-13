from django.contrib import messages
from django.shortcuts import render
from .forms import ImageUploadForm
from .models import PredictedImage
# from .predict import predict_image  # Import function to make predictions

# Create your views here.
# views.py

from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import PredictedImage
from .forms import ImageUploadForm
from PIL import Image as PILImage
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from django.shortcuts import redirect, render
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.contrib.auth import login, logout as django_logout
from django.contrib.auth.decorators import login_required

# from app1.forms import LungImageForm


# Create your views here.
def home(request):
    if request.user == 'AnonymousUser':
        return redirect('login')
    return render(request,'index.html')

def user(request):
    return render(request,'user.html')


def login_signup(request):
    if request.method == 'POST':
        if request.POST.get('action') == 'signup':
            signup_form = UserCreationForm(request.POST)
            entemail = request.POST.get('email')
            entpassword = request.POST.get('password1')
            print(entpassword)
            print(entemail)
            if signup_form.is_valid():
                user = signup_form.save()
                login(request, user)
                messages.success(request, 'Registration successful')
                return redirect('login_or_signup')  # Replace 'home' with your home page URL
        elif request.POST.get('action') == 'login':
            login_form = AuthenticationForm(request, data=request.POST)
            password = request.POST.get('password')
            email = request.POST.get('email')
            print(password)
            print(email)    
            if login_form.is_valid():
                user = login_form.get_user()
                login(request, user)
                return redirect('user')
            else :
                messages.warning(request, 'Invalid Credentials')# Replace 'home' with your home page URL
                return redirect('login_or_signup')
        else :
            return HttpResponse('Not working')

    # Handle GET request or invalid form submissions
    signup_form = UserCreationForm()
    login_form = AuthenticationForm()
    return render(request, 'login.html', {'signup_form': signup_form, 'login_form': login_form})





#Load the pre-trained model
model = load_model('d:\\PDEU\\SEM-2\\NN & DL\\Cancer-Detection-Website\\app2\\best_inception_model.h5')
print("*** >Lung Cancer Model Loaded Successfully< ***")




def upload_image(request):
    if request.method == 'POST':
        print("1")
        form = ImageUploadForm(request.POST, request.FILES)
        if form.is_valid():
            new_image = PredictedImage(image=request.FILES['image'])
            new_image.save()
            print("2")
            # Load and preprocess the uploaded image
            img = PILImage.open(new_image.image)
            img = img.resize((299, 299))  # Assuming the model input size is (299, 299)
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            # Make a prediction using the model
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)

            # Map predicted class to category
            categories = ['Benign Cases', 'Malignant Cases', 'Normal Cases']
            predicted_category = categories[predicted_class]

            # Save the prediction result to the model
            new_image.result = predicted_category
            new_image.save()

            return redirect('result')   # Redirect to the result page
    else:
        form = ImageUploadForm()
        print("3")
    return render(request, 'om_upload.html', {'form': form})

def result(request):
    latest_image = PredictedImage.objects.last()
    return render(request, 'result.html', {'latest_image':latest_image})

def logout(request):
    django_logout(request)  
    return redirect('home')



# from .forms import ColonUploadForm
# from .models import UploadedImage
# import numpy as np
# from tensorflow.keras.models import load_model
# from tensorflow.keras.preprocessing import image

# # Load the pre-trained model
# model = load_model('C:\\Users\\Dhruv Patel\\Desktop\\bisag_django\\project\\cancer\\app1\\Coloncancer1_model.keras')

# # Define the target size for image resizing (must match the input size used during training)
# from django.shortcuts import render
# from .forms import ColonUploadForm
# from .models import UploadedImage
# import numpy as np
# from PIL import Image
# from keras.preprocessing import image
# # Import your trained model here if not already imported

# # Define the target size for the image
# target_size = (299, 299)


# from PIL import Image as PILImage
# import numpy as np
# from keras.preprocessing import image
# from .forms import ColonUploadForm
# from .models import UploadedImage
# from django.shortcuts import render

# # Define the predict_image function
# def predict_image(image_arr):
#     try:
#         print('101')
#         processed_img = image_arr
#         print('102')
#         if processed_img is not None:
#             # Assuming model is defined globally or imported from somewhere
#             print('103')
#             # Make prediction
#             prediction = model.predict(processed_img)
#             print('104')
#             # Get the predicted class (0 for Benign, 1 for Malignant)
#             predicted_class = np.argmax(prediction, axis=1)[0]
#             print('105')
#             # Define your classes
#             classes = ['Benign Cases', 'Malignant Cases']
#             # Return the result and probability
#             print('106')
#             probability = float(prediction[0][predicted_class])
#             print('107')
#             result = classes[predicted_class]
#             print('108')
#             return result, probability
#         else:
#             print("Error: Image preprocessing failed")
#             return 'Error: Image preprocessing failed', 0.0
#     except Exception as e:
#         print(f'Error during prediction: {str(e)}')
#         return f'Error during prediction: {str(e)}', 0.0

# # Define the colon_cancer view
# def colon_cancer(request):
#     if request.method == 'POST':
#         form = ColonUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             print('1')
#             new_image = UploadedImage(image=request.FILES['image'])
#             new_image.save()
#             # Load and preprocess the uploaded image
#             img = PILImage.open(new_image.image)
#             print('2')
#             img = img.resize((299, 299))  # Assuming the model input size is (299, 299)
#             print('3')
#             img_array = image.img_to_array(img)
#             print('4')
#             img_array = np.expand_dims(img_array, axis=0)
#             print('5')
#             img_array /= 255.0
#             print('6')
#             # Call predict_image and get the result and probability
#             result, probability = predict_image(image_arr=img_array)
#             print('7')
#             # Pass the result and probability to the template context
#             return render(request, 'result_colon.html', {'result': result, 'probability': probability})
#     else:
#         form = ColonUploadForm()
#     return render(request, 'upload_colon.html', {'form': form})



# from PIL import Image as PILImage  # Import PIL's Image module as PILImage
# import numpy as np
# from keras.preprocessing import image
# from .forms import ColonUploadForm
# from .models import UploadedImage
# from django.shortcuts import render

# # Define the predict_image function
# def predict_image(image_arr):
#     try:
#         print('101')
#         processed_img = image_arr
#         print('102')
#         if processed_img is not None:
#             # Make prediction
#             prediction = model.predict(processed_img)
#             print('103')
#             # Get the predicted class (0 for Benign, 1 for Malignant)
#             predicted_class = np.argmax(prediction, axis=1)[0]
#             print('104')
#             # Define your classes
#             classes = ['Benign Cases', 'Malignant Cases']
#             # Return the result and probability
#             print('105')
#             probability = float(prediction[0][predicted_class])
#             print('106')
#             result = classes[predicted_class]
#             print('107')
#             return result, probability
#         else:
#             print("Error: Image preprocessing failed")
#             return 'Error: Image preprocessing failed', 0.0
#     except Exception as e:
#         print(f'Error during prediction: {str(e)}')
#         return f'Error during prediction: {str(e)}', 0.0

# # Define the colon_cancer view
# def colon_cancer(request):
#     if request.method == 'POST':
#         form = ColonUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             print('1')
#             new_image = UploadedImage(image=request.FILES['image'])
#             new_image.save()
#             # Load and preprocess the uploaded image
#             img = PILImage.open(new_image.image)
#             print('2')
#             img = img.resize((299, 299))  # Assuming the model input size is (299, 299)
#             print('3')
#             img_array = image.img_to_array(img)
#             print('4')
#             img_array = np.expand_dims(img_array, axis=0)
#             print('5')
#             img_array /= 255.0
#             print('6')
#             # Call predict_image and get the result and probability
#             result, probability = predict_image(image_arr=img_array)
#             print('7')
#             # Pass the result and probability to the template context
#             return render(request, 'result_colon.html', {'result': result, 'probability': probability})
#     else:
#         form = ColonUploadForm()
#     return render(request, 'upload_colon.html', {'form': form})





# from django.shortcuts import render
# from django.http import JsonResponse
# from .utils import predict_image

# def colon_cancer(request):
#     if request.method == 'POST':
#         print('1')
#         form = ColonUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             print('2')
#             # Save the uploaded image to the database
#             uploaded_image = form.save(commit=False)
#             print('3')
#             uploaded_image.result, uploaded_image.probability = predict_image(uploaded_image.image)
#             print('4')
#             uploaded_image.save()
#             return render(request, 'result_colon.html', {'uploaded_image': uploaded_image})
#     else:
#         form = ColonUploadForm()
#     return render(request, 'upload_colon.html', {'form': form})
# def colon_cancer(request):
#     if request.method == 'POST':
#         form = ColonUploadForm(request.POST, request.FILES)
#         if form.is_valid():
#             image = form.cleaned_data['image']
#             prediction, probability = predict_image(image)
#             return render(request, 'result_colon.html', {'prediction': prediction, 'probability': probability})
#     else:
#         form = ColonUploadForm()
#     return render(request, 'upload_colon.html', {'form': form})
