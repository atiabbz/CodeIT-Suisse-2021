from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("stringTest", views.stringTest, name="stringTest"),
    path("jsonTest", views.jsonTest, name="jsonTest"),
    path("consumeJson", views.consumeJson, name="consumeJson"),
    path("asteroid", views.asteroid, name="asteroid"),
    path("optopt", views.calculate, name="optopt"),
    path("parasite", views.evaluateParasite, name="parasite"),
    path("stock-hunter", views.stockHunting, name="stock-hunter"),
    path("cipher-cracking", views.cipher_cracking, name="cipher-cracking"),
    path("stig/perry", views.evaluateInterviews, name="stig-perry"),
]
