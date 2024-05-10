from django.db import models

# Create your models here.

class UploadedImage(models.Model):
    image = models.ImageField(upload_to='images/')
    result = models.CharField(max_length=255, null=True)
    probability = models.FloatField(null=True)

    def __str__(self):
        return f"Uploaded Image {self.pk}"