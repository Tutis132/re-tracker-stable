
import os
import time
import random
import datetime
import logging
import atexit
import shutil

from selenium import webdriver
from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options

import options

driver = None
log = logging.getLogger('estate.aruodas')
pid = os.getpid()


def init_userdir(create=True):
    userdir = f'/tmp/estate-chrome.{pid}'
    if os.path.exists(userdir):
        shutil.rmtree(userdir)
    if create:
        os.mkdir(userdir)
    return userdir


def find_chrome_browser():
    if os.path.isfile('/usr/bin/google-chrome-stable'):
        return '/usr/bin/google-chrome-stable'
    elif os.path.isfile('/usr/bin/google-chrome'):
        return '/usr/bin/google-chrome'
    elif os.path.isfile('/usr/bin/chrome'):
        return '/usr/bin/chrome'
    elif os.path.isfile('/usr/bin/chromium-browser'):
        return '/usr/bin/chromium-browser'
    elif os.path.isfile('/usr/bin/chromium'):
        return '/usr/bin/chromium'
    else:
        return None


def exit_selenium():
    if not driver:
        return
    time.sleep(2)
    driver.quit()
    init_userdir(False)


def init_selenium():
    global driver

    user_agent = 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36'

    #user_dir = init_userdir()

    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--window-size=1920,1080")
    chrome_options.add_argument(f"--user-agent={user_agent}")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-extensions")
    chrome_options.add_argument("--disable-infobars")
    chrome_options.add_argument('--disable-dev-shm-usage')
    #chrome_options.add_argument(f"--user-data-dir={user_dir}")
    chrome_options.add_experimental_option(
        'excludeSwitches', ['enable-logging'])
    chrome_exec = find_chrome_browser()
    if chrome_exec:
        chrome_options.binary_location = chrome_exec
    else:
        log.warning("Cannot find system chrome browser")

    driver = webdriver.Chrome(options=chrome_options)

    atexit.register(exit_selenium)


def high_list(item):
    try:
        item.find_element(By.CLASS_NAME, "icon-house-list")
        return True
    except NoSuchElementException:
        return False


class AruodasPage:

    def __init__(self, listing):
        self.listing = listing
        self.items = None
        self.index = None

    def __iter__(self):
        try:
            self.items = self.listing.find_elements(By.CLASS_NAME, "object-row")
            self.index = 0
        except NoSuchElementException as e:
            log.error("Error getting places '%s'", driver.current_url)
            if options.debug:
                content = self.listing.get_attribute('innerHTML')
                print(f"Invalid page payload:\n{content}")
                raise e
        return self

    def __next__(self):
        if not self.items or self.index >= len(self.items):
            raise StopIteration

        item = self.items[self.index]
        self.index += 1
        log.debug("Retrieve %i/%i on %s", self.index, len(self.items), driver.current_url)
        return item


class AruodasNavigator:

    def __init__(self):
        self.init_url = None
        self.curr_url = None

    def __grab_listing(self, url):
        driver.get(url)
        try:
            page_list = driver.find_element(By.CLASS_NAME, "list-search-v2")
            log.debug("Grabbed aruodas URL '%s'", url)
            return AruodasPage(page_list)
        except NoSuchElementException as e:
            try:
                driver.find_element(By.ID, "challenge-form")
                log.error("Cought in reCAPCHA trap: '%s'", url)
            except NoSuchElementException:
                log.error("Error getting search '%s'", url)
            if options.debug:
                content = driver.find_element(By.XPATH, "/*").get_attribute('innerHTML')
                print(f"Invalid content payload:\n{content}")
                raise e
            return None

    def init_page(self, url):
        self.init_url = url
        self.curr_url = url
        return self.__grab_listing(url)

    def next_page(self):
        
        try:
            pages = driver.find_element(By.CLASS_NAME, "pagination").find_elements(By.TAG_NAME, "a")
        except NoSuchElementException as e:
            print(e)
            print("Pagination element not found")
            return None

        page_next = pages[-1]

        if 'page-bt-disabled' in page_next.get_attribute('class').split():
            log.debug("Final page reached")
            return None

        self.curr_url = page_next.get_attribute("href")
        if not self.curr_url:
            log.error("Invalid next page URL")
            return None

        return self.__grab_listing(self.curr_url)


class AruodasSurfer:

    def __init__(self, journal, record_class):
        grab_now = datetime.datetime.now()
        self.grab_time = grab_now.strftime("%Y-%m-%d %H:%M:%S")
        self.grab_page = 0
        self.journal = journal
        self.record_class = record_class
        self.journal.mark(self.grab_time)
        self.high_list_pass = True

    def grab(self, page):
        rec_received = 0
        rec_checked = 0
        rec_updated = 0
        self.grab_page += 1
        for item in page:
            # Mark when we have passed Highlighted
            # adverts to early terminate surfing.
            if self.high_list_pass:
                self.high_list_pass = high_list(item)
                if not self.high_list_pass:
                    log.info("Highlight absent in page %i", self.grab_page)
            record = self.record_class(item)
            # Record grab session time
            record.time = self.grab_time
            log.debug("Inspecting record '%s'", record)
            rec_received += 1

            rec_checked += 1
            if self.journal.define(record):
                rec_updated += 1

        if rec_checked == 0:
            log.info("Nothing accepted in %i page", self.grab_page)
            return self.high_list_pass
        else:
            log.info("Accepted %i/%i records in %i page",
                     rec_updated, rec_received, self.grab_page)
            return True

    def surf(self, url):
        log.info("Surfing '%s'", url)
        aruodas = AruodasNavigator()
        page = aruodas.init_page(url)
        if not page:
            log.error("Nothing found in %s", url)
            return
        while page and self.grab(page):
            if options.onerun:
                break
            delay = random.randrange(10, 20)
            log.debug("Proceeding to next page after %i seconds", delay)
            time.sleep(delay)
            page = aruodas.next_page()
