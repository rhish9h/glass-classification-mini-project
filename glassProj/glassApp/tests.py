# from django.test import TestCase

# Create your tests here.
from selenium import webdriver
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse
import time
import random
from selenium.webdriver.common.keys import Keys

class SelTest(StaticLiveServerTestCase):
  def setUp(self):
    self.browser = webdriver.Firefox()
    super(SelTest, self).setUp()

  def tearDown(self):
    return self.browser.quit()
    super(SelTest, self).tearDown()
  
  def random_value_form_check(self):
    # self.browser.get('http://localhost:8000') # open website
    # time.sleep(0.5)
    elements = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']
    low_limits = [1.51, 10.7, 0, 0.29, 69.8, 0, 5.43, 0, 0]
    upper_limits = [1.53, 17.4, 4.49, 3.5, 75.4, 6.21, 16.2, 3.15, 0.51]
    for i in range(len(elements)):
      ri = self.browser.find_element_by_id(elements[i]) # send key to ri
      ri.click() # focus on input box
      ri.send_keys(str(random.uniform(low_limits[i], upper_limits[i]))) # send random float value in range 0, 100 both inclusive, depending on rounding
    time.sleep(0.5)
    self.browser.find_element_by_id('submit_button').click() # click submit button
    time.sleep(2)
  
  def about_page_linking(self):
    self.browser.get('http://localhost:8000') # open website
    time.sleep(0.5)
    self.browser.find_element_by_id('about_button').click() # click on about button
    time.sleep(0.5)
    self.browser.find_element_by_id('about_button').click() # click on about button
    time.sleep(0.5)
    self.browser.find_element_by_id('glass_heading').click() # click on home button
    time.sleep(0.5)
    self.browser.find_element_by_id('about_button').click() # click on about button
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('glass_heading').click() # click on home button
    time.sleep(0.5)

  def graph_page_linking(self):
    self.browser.maximize_window()
    self.browser.get('http://localhost:8000') # open website
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('glass_heading').click() # click on home button
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('about_button').click() # click on about button
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    ri = self.browser.find_element_by_id('kmeansK') # send key to ri
    ri.click() # focus on input box
    ri.send_keys(str(random.randint(1, 10))) # send random float value in range 0, 100 both inclusive, depending on rounding
    self.browser.find_element_by_id('genKmeans_button').click() # click on genKmeans button
    time.sleep(2)
    self.browser.find_element_by_id('glass_heading').click() # click on home button
    time.sleep(0.5)

  def test_all_together(self):
    self.graph_page_linking()
    self.random_value_form_check()