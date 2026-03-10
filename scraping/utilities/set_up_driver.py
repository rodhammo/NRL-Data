"""
Module to set up Chrome WebDriver for scraping.

Uses webdriver-manager to automatically download and cache the correct
ChromeDriver binary for the installed Chrome version.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


def set_up_driver():
    """Set up a headless Chrome WebDriver for scraping.

    :return: WebDriver object for Chrome
    """
    options = Options()
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--headless=new')
    options.add_argument('--log-level=3')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    service = Service(ChromeDriverManager().install())
    driver = webdriver.Chrome(service=service, options=options)
    return driver
