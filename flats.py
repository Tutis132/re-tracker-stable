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

log = logging.getLogger('estate.flats')


class FlatRecord(Table):
    """
        self.time = spot['time']
        self.url = spot['url']
        self.banner = spot['banner']
        self.address = spot['address']
        self.promo = spot['promo']
        self.price = spot['price']
        self.square = spot['square']
        self.home_size = spot['home_size']
        self.home_rooms = spot['home_rooms']
        self.home_floor = spot['home_floor']
        self.floor_count = spot['floor_count']

        spot = {
            'time': self.time,
            'url': self.url,
            'banner': self.banner,
            'address': self.address,
            'promo': self.promo,
            'price': self.price,
            'square': self.square,
            'home_size': self.home_size,
            'home_rooms': self.home_rooms,
            'home_floor': self.home_floor,
            'floor_count': self.floor_count,
        }
    """

    def __init__(self, item=None):
        if item:
            self.parse(item)

    def __str__(self):
        return f"{self.address}, {self.home_size}m2 {self.home_rooms}r {self.home_floor}f"

    def __eq__(self, other):
        return \
                self.address == other.address and \
                self.state == other.state and \
                self.home_size == other.home_size and \
                self.home_rooms == other.home_rooms and \
                self.home_floor == other.home_floor and \
                self.floor_count == other.floor_count and \
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
            price = advert.find_element(By.CLASS_NAME, "list-item-price-v2").text
            square = advert.find_element(By.CLASS_NAME, "price-pm-v2").text
            home_size = advert.find_element(By.CLASS_NAME, "list-AreaOverall-v2").text.strip()
            home_rooms = advert.find_element(By.CLASS_NAME, "list-RoomNum-v2").text.strip()
            block_floor = advert.find_element(By.CLASS_NAME, "list-Floors-v2").text.strip()

            home_floor, sep, floor_count = block_floor.partition('/')

            record['url'] = url
            record['banner'] = banner
            record['address'] = address.strip().replace('\n', ',')
            record['price'] = float(re.search(r'(\d+)', price.strip().replace('\n', '').replace(' ', '')).group(1))
            record['square'] = float(re.search(r'(\d+)', square.strip().replace('\n', '').replace(' ', '')).group(1))
            record['home_size'] = float(home_size)
            record['home_rooms'] = int(home_rooms)
            record['home_floor'] = int(home_floor)
            record['floor_count'] = int(floor_count)
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

        log.info("Parsed addr:'%s', promo:'%s', price:'%s', square:'%s', home_size:'%s', home_rooms:'%s', home_floor:'%s' floor_count:'%s'",
                  record['address'], record['promo'], record['price'], record['square'],
                  record['home_size'], record['home_rooms'], record['home_floor'], record['floor_count'])

        self.update(record)


def run_flat_scrape(section):
    print(f'section: {section}')
    options.Params(section=section)
    print(f'table name: {options.table}')
    print(f'type: {options.type}')
    aruodas.init_selenium()
    database = PropertyJournal(options.table, options.type, FlatRecord)
    houses = AruodasSurfer(database, FlatRecord)
    houses.surf(options.url)
    database.save()

if __name__ == "__main__":
    run_flat_scrape()
