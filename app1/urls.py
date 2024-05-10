from django.urls import path
from . import views
from django.urls import include, path
from . import views

urlpatterns = [
    path('',views.home, name='home'),
    path('login/', views.login_signup, name='login_or_signup'),
    path('user/', views.user, name='user'),
    path('logout/', views.logout, name='logout'),
    path('upload/', views.upload_image, name='upload_image'),
    path('result/', views.result, name='result'),
]