from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle
import os

# Create your views here.

modulePath = os.path.dirname(__file__)
# ----------------------------------------------------------------------------------------knn

data = pd.read_csv(os.path.join(modulePath, 'glass.csv'))
labels = data.pop("Type").values
data = data.values
labels = labels.reshape(-1, 1)

class KNN:
    def __init__(self, k, d_metric, p=1):
        self.k = k
        self.d_metric = d_metric
        self.d_metric_to_fn = {
            'euclidean': self.euclidean,
        }
        self.p = p

    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)

    def euclidean(self, x_test):
        sq_diff = (self.X - x_test) ** 2
        return np.sqrt(np.sum(sq_diff, axis=-1))
 
    def distance(self, x_test):
        return self.d_metric_to_fn[self.d_metric](x_test)

    def predict(self, x_test):
        distances = self.distance(x_test)
        sorted_labels = self.y[np.argsort(distances)]
        k_sorted_labels = sorted_labels[:self.k]
        unique_labels, counts = np.unique(k_sorted_labels, return_counts=True)
        pred1 = unique_labels[np.argmax(counts)]
        return pred1
    
    
    def predict2(self, x_test): #for acc
        preds = []
        for index in range(x_test.shape[0]):
            distances = self.distance(x_test[index])
            sorted_labels = self.y[np.argsort(distances)]
            k_sorted_labels = sorted_labels[:self.k]
            unique_labels, counts = np.unique(k_sorted_labels, return_counts=True)
            pred = unique_labels[np.argmax(counts)]
            preds.append(pred)
        return np.array(preds)
    
    def accuracy(self, data, labels):
        pred = self.predict2(data)
        count = 0
        for i in range(len(pred)):
            if pred[i] == labels[i]:
                count += 1
        return float(count)/len(pred)

# ----------------------------------------------------------------------------------------knn end

# load knn model pickle
# knn_path = os.path.join(modulePath, 'knn_pickle')

# with open(knn_path, 'rb') as file:
#   knn_model = pickle.load(file)
# pickle load end

# create model
knn = KNN(k=3, d_metric='euclidean')
knn.fit(data, labels)

# ------------------------------------------------------------------------index function
def index(request):
  if request.method == 'POST': # --------------------------- post -------
    form_data = {
      'ri': request.POST.get('ri', ''),
      'na': request.POST.get('na', ''),
      'mg': request.POST.get('mg', ''),
      'al': request.POST.get('al', ''),
      'si': request.POST.get('si', ''),
      'k': request.POST.get('k', ''),
      'ca': request.POST.get('ca', ''),
      'ba': request.POST.get('ba', ''),
      'fe': request.POST.get('fe', ''),
    }

    # convert form data to array of integers for processing in model
    ip_values = []
    for key, value in form_data.items():
      try: # for blank values
        ip_values.append(float(value))
      except:
        ip_values.append(0)

    # predict values
    glassType = {
      1: 'building_windows_float_processed',
      2: 'building_windows_non_float_processed', 
      3: 'vehicle_windows_float_processed', 
      4: 'vehicle_windows_non_float_processed', 
      5: 'containers', 
      6: 'tableware', 
      7: 'headlamps'
    }
    
    prediction_key = knn.predict([ip_values])
    prediction = glassType[prediction_key]

    return render(request, 'glassApp/index.html', {'form_data': form_data, 'ip_values': ip_values, 'prediction': prediction})
    #  --------------------------------------------------------------------------------------get
  else:
    return render(request, 'glassApp/index.html')

def graphs(request):
  return render(request, 'glassApp/graphs.html')