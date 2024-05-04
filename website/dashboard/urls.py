from django.urls import path
from . import views

urlpatterns = [
    path('model_performance/', views.model_performance, name='model_performance'),
]