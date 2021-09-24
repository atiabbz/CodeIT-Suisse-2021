from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stringTest", views.stringTest, name="stringTest"),
    path("jsonTest", views.jsonTest, name="jsonTest"),
    path("consumeJson", views.consumeJson, name="consumeJson"),
]
