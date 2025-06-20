from django.urls import path
from .views import predict_fake_news, home

urlpatterns = [
    path('', home, name='home'),
    path('predict/', predict_fake_news, name='predict_fake_news'),
] 