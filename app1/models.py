from django.db import models

# Create your models here.

class PredictedImage(models.Model):
    image = models.ImageField(upload_to='uploaded_images/')
    result = models.CharField(max_length=500)


      
