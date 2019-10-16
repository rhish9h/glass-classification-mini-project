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
  
  # def test_foo(self):
  #   self.assertEquals(0, 1)

  def graph_page_linking(self):
    self.browser.get('http://localhost:8000') # open website
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('graph_button').click() # click on graph button
    time.sleep(0.5)
    self.browser.find_element_by_id('glass_heading').click() # click on home button
    time.sleep(0.5)
  
  def random_value_form_check(self):
    # self.browser.get('http://localhost:8000') # open website
    # time.sleep(0.5)

    elements = ['ri', 'na', 'mg', 'al', 'si', 'k', 'ca', 'ba', 'fe']

    for i in elements:
      ri = self.browser.find_element_by_id(i) # send key to ri
      ri.click() # focus on input box
      ri.send_keys(str(random.uniform(0, 100))) # send random float value in range 0, 100 both inclusive, depending on rounding

    time.sleep(0.5)
    self.browser.find_element_by_id('submit_button').click() # click submit button
    time.sleep(2)

  def test_all_together(self):
    self.graph_page_linking()
    self.random_value_form_check()