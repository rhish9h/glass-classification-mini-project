from django.shortcuts import render

# Create your views here.

def index(request):
  return render(request, 'glassApp/index.html')

def graphs(request):
  return render(request, 'glassApp/graphs.html')