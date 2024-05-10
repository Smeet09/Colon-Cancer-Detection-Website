from django.db import models

# Create your models here.

class PredictedBreastCancer(models.Model):
    image = models.ImageField(upload_to='breast_cancer/')
    result = models.CharField(max_length=500)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Prediction: {self.result} - {self.created_at}"
