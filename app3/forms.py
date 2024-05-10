from django import forms

class BreastCancerImageUploadForm(forms.Form):
    image = forms.ImageField(label='Select an image file')
