import unittest
from cal import KNN

class Testcal(unittest.TestCase):

  def test_predict(self):
      cal = KNN(3, 'euclidean')
      result=cal.predict([[1.51514,14.85,0,2.42,73.72,0,8.39,0.56,0]])
      self.assertEqual(result,7)

  def test_accur(self):
      cal = KNN(3, 'euclidean')
      result=cal.accur('euclidean')
      self.assertEqual(result,0.83644)

  def test_result_knn(self):
      cal = KNN(3, 'euclidean')
      result=cal.result_knn('euclidean')
      self.assertEqual(result,2)

if __name__== '__main__':
  unittest.main()