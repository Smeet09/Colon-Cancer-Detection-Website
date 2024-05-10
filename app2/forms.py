from django import forms
from .models import *

class ColonUploadForm(forms.ModelForm):
    class Meta:
        model = UploadedImage
        fields = ['image']