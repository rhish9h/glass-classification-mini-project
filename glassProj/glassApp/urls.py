from django.urls import path

from . import views

urlpatterns = [
    path('', views.index, name='index'),
    path('index', views.index, name='index'),
    path('graphs', views.graphs, name='graphs'),
    path('genKmeans', views.genKmeans, name='genKmeans'),
    path('about', views.about, name='about'),
]
