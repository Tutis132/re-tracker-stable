#!/usr/bin/env python

import re
import logging
import logging.config

from utils import Table

from selenium.common.exceptions import NoSuchElementException
from selenium.webdriver.common.by import By

logging.config.fileConfig('logging.ini')

import options
import aruodas
from utils import download_file
from storage import PropertyJournal
from aruodas import AruodasSurfer

log = logging.getLogger('estate.houses')


class HouseRecord(Table):
    """
        self.time = spot['time']
        self.url = spot['url']
        self.banner = spot['banner']
        self.address = spot['address']
        self.promo = spot['promo']
        self.price = spot['price']
        self.square = spot['square']
        self.home_size = spot['home_size']
        self.site_size = spot['site_size']
        self.state = spot['state']

        spot = {
            'time': self.time,
            'url': self.url,
            'banner': self.banner,
            'address': self.address,
            'promo': self.promo,
            'price': self.price,
            'square': self.square,
            'home_size': self.home_size,
            'site_size': self.site_size,
            'state': self.state,
        }
    """

    def __init__(self, item=None):
        if item:
            self.parse(item)

    def __str__(self):
        return f"{self.address}, {self.state}, {self.home_size}m2 {self.site_size}a"

    def __eq__(self, other):
        return \
                self.address == other.address and \
                self.state == other.state and \
                self.home_size == other.home_size and \
                self.site_size == other.site_size and \
                self.price == other.price

    def __ne__(self, other):
        return not self.__eq__(other)

    def save(self, file_path):
        if download_file(self.banner, f"{file_path}/{self.time}.jpg"):
            self.banner_file = f"{self.time}.jpg"
        else:
            log.error("Error getting banner for '%s'", self.url)
        super().save(file_path)

    def parse(self, item):
        record = {}

        try:
            advert = item.find_element(By.CLASS_NAME, "advert-flex")
        except Exception as e:
            # Fallback use root item
            advert = item
            log.error("Error getting place root container")
            if options.debug:
                content = item.get_attribute('innerHTML')
                print(f"Invalid place params:\n{content}")
                raise e

        url = ""
        try:
            url = advert.find_element(By.TAG_NAME, "a").get_attribute("href")
            banner = advert.find_element(By.TAG_NAME, "img").get_attribute("src")
            address = advert.find_element(By.CLASS_NAME, "list-adress-v2").find_element(By.TAG_NAME, "h3").text
            state = advert.find_element(By.CLASS_NAME, "list-HouseStates-v2").text
            price = advert.find_element(By.CLASS_NAME, "list-item-price-v2").text
            square = advert.find_element(By.CLASS_NAME, "price-pm-v2").text
            home_size = advert.find_element(By.CLASS_NAME, "list-AreaOverall-v2").text.strip()
            site_size = advert.find_element(By.CLASS_NAME, "list-AreaLot-v2").text.strip()

            record['url'] = url
            record['banner'] = banner
            record['address'] = address.strip().replace('\n', ',')
            record['state'] = state.strip()
            record['price'] = float(re.search(r'(\d+)', price.strip().replace('\n', '').replace(' ', '')).group(1))
            record['square'] = float(re.search(r'(\d+)', square.strip().replace('\n', '').replace(' ', '')).group(1))
            record['home_size'] = float(home_size)
            # XXX: there were cases when house site size is not specified.
            # Use 0 in this case to indicate undefined site size.
            record['site_size'] = float(site_size if site_size != '' else 0)
        except Exception as e:
            log.error("Error getting place params: %s", url)
            if options.debug:
                content = item.get_attribute('innerHTML')
                print(f"Invalid place params:\n{content}")
                raise e

        try:
            promo = advert.find_element(By.CLASS_NAME, "price-change").text.strip()
            key, val = re.search(r"Kaina\s+([ps]).*\s+([\d\,]+)%$", promo).groups()
            record['promo'] = float(val.replace(',', '.'))
            if key == 's':
                record['promo'] *= -1
        except NoSuchElementException:
            record['promo'] = 0
        except (TypeError, ValueError) as e:
            log.error("Error parsing place promo: '%s'", url)
            if options.debug:
                content = item.get_attribute('innerHTML')
                print(f"Invalid place promo:\n{content}")
                raise e

        try:
            record['reserved'] = advert.find_element(By.CLASS_NAME, "reservation-strip") and True
        except NoSuchElementException:
            record['reserved'] = None

        log.info("Parsed addr:'%s', state:'%s', promo:'%s', price:'%s', square:'%s', home_size:'%s', site_size:'%s'",
                  record['address'], record['state'], record['promo'], record['price'],
                  record['square'], record['home_size'], record['site_size'])

        self.update(record)


def run_house_scrape(section):
    options.Params(section=section)
    aruodas.init_selenium()
    database = PropertyJournal(options.table, options.type, HouseRecord)
    houses = AruodasSurfer(database, HouseRecord)
    houses.surf(options.url)
    database.save()

if __name__ == "__main__":
    run_house_scrape()
