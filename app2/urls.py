from django.urls import path
from . import views
urlpatterns = [
    path('colon-cancer/', views.colon_cancer, name='colon'),
]