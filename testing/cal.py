import pandas as pd 
import numpy as np 
import os

for dirname, _, filenames in os.walk('glass.csv'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# modulePath = os.path.dirname(__file__)
# data = pd.read_csv(os.path.join(modulePath, 'glass.csv'))



class KNN:
    def __init__(self, k, d_metric, p=1):
        self.k = k
        self.d_metric = d_metric
        self.d_metric_to_fn = {
            'euclidean': self.euclidean,
            
        }
        self.p = p
        self.data = pd.read_csv("glass.csv")
        self.labels = self.data.pop("Type").values
        self.data = self.data.values
        self.labels = self.labels.reshape(-1, 1)
        self.fit(self.data, self.labels)


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
    
    
    def predict2(self, x_test):
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
 
 
    def result_knn(self, metric):
  

        # knn = KNN(k=3, d_metric=metric)
        # knn.fit(data, labels)

        # prediction = knn.predict(data[100,:])
        prediction = self.predict(self.data[100,:])

        return prediction

    def accur(self, metric):
        accuraci = {}

        for k in range(1, 21):
            knn = KNN(k=k, d_metric=metric)
            knn.fit(self.data, self.labels)
        
            accuraci[k] = knn.accuracy(self.data, self.labels)
        
        return accuraci[3]
    