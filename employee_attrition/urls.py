
from django.urls import path
from . import views

urlpatterns = [
    path('', views.predict_attrition, name='predict_attrition'),
    path('result/', views.predict_attrition, name='predict_result'),
]
