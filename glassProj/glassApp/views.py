from django.shortcuts import render
import numpy as np
import pandas as pd
import pickle
import os
from django.http import HttpResponse

# Create your views here.

modulePath = os.path.dirname(__file__)
# ----------------------------------------------------------------------------------------knn

data = pd.read_csv(os.path.join(modulePath, 'glass_os.csv'))
# Remove first columns as index base 
data.drop(data.columns[[0]], axis = 1, inplace = True)
labels = data.pop("Type").values
data = data.values
labels = labels.reshape(-1, 1)

class KNN:
    def __init__(self, k, d_metric, p=1):
        self.k = k
        self.d_metric = d_metric
        self.d_metric_to_fn = {
            'euclidean': self.euclidean,
            'manhattan': self.manhattan,
            'minkowski': self.minkowski
        }
        self.p = p

    def fit(self, X, y):
        self.X = np.copy(X)
        self.y = np.copy(y)

    def manhattan(self, x_test):
        return np.sum(np.abs(self.X - x_test), axis=-1)

    def euclidean(self, x_test):
        sq_diff = (self.X - x_test) ** 2
        return np.sqrt(np.sum(sq_diff, axis=-1))
 
    def minkowski(self, x_test):
        abs_diff = np.abs(self.X - x_test)
        sum_p_diff = np.sum(abs_diff ** self.p, axis=-1)
        pth_root = sum_p_diff ** (1 / self.p)
        return pth_root

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
knn = KNN(k=3, d_metric='manhattan')
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

    default_values = {
      'ri': 1.51,
      'na': 10.7,
      'mg': 0,
      'al': 0.29,
      'si': 69.8,
      'k': 0,
      'ca': 5.43,
      'ba': 0,
      'fe': 0,
    }

    # form_data['ri'] = form_data['ri'] if not form_data['ri'] == '' else 1.51

    # convert form data to array of integers for processing in model
    ip_values = []
    for key, value in form_data.items():
      form_data[key] = form_data[key] if not form_data[key] == '' else default_values[key]
      ip_values.append(float(form_data[key]))

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

def draw2dgraph(kvalue):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import random as rd

  df = pd.read_csv(os.path.join(modulePath, 'glass.csv'))
  X = df.iloc[:, [3, 4]].values
  # print (X)

  m = X.shape[0]  # number of training examples
  n = X.shape[1]  # number of features. Here n=2n_iter=100

  K = 7 if kvalue <= 0 else kvalue
  n_iter = 100
  # Centroids = 0
  Centroids = np.array([]).reshape(n, 0)

  for i in range(K):
      rand = rd.randint(0, m - 1)
      Centroids = np.c_[Centroids, X[rand]]

  Output = {}

  EuclidianDistance = np.array([]).reshape(m, 0)
  for k in range(K):
      tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
      EuclidianDistance = np.c_[EuclidianDistance, tempDist]
  C = np.argmin(EuclidianDistance, axis=1) + 1

  for i in range(n_iter):
      # step 2.a
      EuclidianDistance = np.array([]).reshape(m, 0)
      for k in range(K):
          tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
          EuclidianDistance = np.c_[EuclidianDistance, tempDist]
      C = np.argmin(EuclidianDistance, axis=1) + 1
      # step 2.b
      Y = {}
      for k in range(K):
          Y[k + 1] = np.array([]).reshape(2, 0)
      for i in range(m):
          Y[C[i]] = np.c_[Y[C[i]], X[i]]

      for k in range(K):
          Y[k + 1] = Y[k + 1].T

      for k in range(K):
          Centroids[:, k] = np.mean(Y[k + 1], axis=0)
      Output = Y

  color = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']

  if kvalue > 7: #create colors list for more than 7 K
    number_of_colors = kvalue
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

  # labels = ['bw_fp', 'bw_nfp', 'v', 'v_nfp', 'containers', 'tableware', 'headlamps']
  # ------------------------------------------------------------------------------------2d

  plt.clf()
  for k in range(K):
      plt.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k])
  plt.scatter(Centroids[0, :], Centroids[1, :], s=100, c='grey', label='Centroids')
  plt.xlabel('RI')
  plt.ylabel('Na')
  plt.savefig('assets/images/2dbuf', dpi = 150)

def draw3dgraph(kvalue):
  import pandas as pd
  import numpy as np
  import matplotlib.pyplot as plt
  import random as rd

  df = pd.read_csv(os.path.join(modulePath, 'glass.csv'))

  X = df.iloc[:, [3, 4]].values
  # print (X)

  m = X.shape[0]  # number of training examples
  n = X.shape[1]  # number of features. Here n=2n_iter=100

  K = 7 if kvalue <= 0 else kvalue
  n_iter = 100
  Centroids = np.array([]).reshape(n, 0)

  for i in range(K):
      rand = rd.randint(0, m - 1)
      Centroids = np.c_[Centroids, X[rand]]

  Output = {}

  EuclidianDistance = np.array([]).reshape(m, 0)
  for k in range(K):
      tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
      EuclidianDistance = np.c_[EuclidianDistance, tempDist]
  C = np.argmin(EuclidianDistance, axis=1) + 1

  for i in range(n_iter):
      # step 2.a
      EuclidianDistance = np.array([]).reshape(m, 0)
      for k in range(K):
          tempDist = np.sum((X - Centroids[:, k]) ** 2, axis=1)
          EuclidianDistance = np.c_[EuclidianDistance, tempDist]
      C = np.argmin(EuclidianDistance, axis=1) + 1
      # step 2.b
      Y = {}
      for k in range(K):
          Y[k + 1] = np.array([]).reshape(2, 0)
      for i in range(m):
          Y[C[i]] = np.c_[Y[C[i]], X[i]]

      for k in range(K):
          Y[k + 1] = Y[k + 1].T

      for k in range(K):
          Centroids[:, k] = np.mean(Y[k + 1], axis=0)
      Output = Y

  color = ['red', 'blue', 'green', 'cyan', 'magenta', 'yellow', 'black']
  if kvalue > 7: #create colors list for more than 7 K
    number_of_colors = kvalue
    color = ["#"+''.join([rd.choice('0123456789ABCDEF') for j in range(6)]) for i in range(number_of_colors)]

  # labels = ['bw_fp', 'bw_nfp', 'v', 'v_nfp', 'containers', 'tableware', 'headlamps']
  from mpl_toolkits.mplot3d import Axes3D

  plt.clf()
  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')
  for k in range(K):
      ax.scatter(Output[k + 1][:, 0], Output[k + 1][:, 1], c=color[k])
  ax.scatter(Centroids[0, :], Centroids[1, :], s=100, c='grey', label='Centroids')

  ax.set_xlabel('RI')
  ax.set_ylabel('Na')
  ax.set_zlabel('Mg')
  plt.savefig('assets/images/3dbuf', dpi = 150)

def genKmeans(request):
  kmeansK_ip = request.GET.get('kmeansK', '')
  kmeansK_ip = 7 if kmeansK_ip == '' else int(kmeansK_ip)
  draw2dgraph(kmeansK_ip)
  draw3dgraph(kmeansK_ip)
  return render(request, 'glassApp/genKmeans.html', {'kmeansK_ip': kmeansK_ip})

def about(request):
  return render(request, 'glassApp/about.html')