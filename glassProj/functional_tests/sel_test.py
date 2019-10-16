from selenium import webdriver
from django.contrib.staticfiles.testing import StaticLiveServerTestCase
from django.urls import reverse

class SelTest(StaticLiveServerTestCase):
  def setUp(self):
    self.browser = webdriver.Chrome('functional_tests/chromedriver.exe')
    super(SelTest, self).setUp()

  def tearDown(self):
    return self.browser.quit()
    super(SelTest, self).tearDown()
  
  def test_foo(self):
    self.assertEquals(0, 1)