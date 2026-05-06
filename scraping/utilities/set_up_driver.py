"""
Module to set up Chrome WebDriver for scraping.

Uses Selenium's built-in driver manager to automatically download and cache
the correct ChromeDriver binary for the installed Chrome version.
"""

from selenium import webdriver
from selenium.webdriver.chrome.options import Options


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

    driver = webdriver.Chrome(options=options)
    return driver
