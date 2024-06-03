
import logging
import datetime

from utils import Table

import json

from app_setup import app
from app_setup import db

from db_setup import Session
from models import create_property_model

from sqlalchemy.exc import SQLAlchemyError
log = logging.getLogger('estate.storage')

def update_last_checked_file(table_name):
        # TODO save each day when it was triggered
        filename = 'last_checked.json'
        try:
            with open(filename, 'r') as file:
                last_checked = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError):
            last_checked = {}

        last_checked[table_name] = datetime.datetime.now().isoformat()

        with open(filename, 'w') as file:
            json.dump(last_checked, file, indent=4)

def get_last_checked_date(table_name):
    filename = 'last_checked.json'
    try:
        with open(filename, 'r') as file:
            last_checked = json.load(file)
            return last_checked.get(table_name)
    except FileNotFoundError:
        return None

class PropertyJournal:

    def __init__(self, table_name, property_type, record_class):
        self.record_class = record_class
        self.places = {}
        self.time = None
        self.table_name = table_name
        self.property_type = property_type
        with app.app_context():
            self.Property = create_property_model(table_name, property_type)
            db.create_all()
        self.__load()

    def __load(self):
        # In future, consider optimize by reading database before scraping. This would save some time.
        pass

    def return_property(self, advert):
        property_args = {
            'time': datetime.datetime.now(),
            'url': advert.url,
            'address': advert.address,
            'promo': advert.promo,
            'price': advert.price,
            'square': advert.square,
            'home_size': getattr(advert, 'home_size', None),
            'last_found_date': datetime.datetime.now()
        }

        if self.property_type == 'house':
            property_args.update({
                'site_size': getattr(advert, 'site_size', None),
                'state': advert.state
            })

        if self.property_type == 'flat':
            property_args.update({
                'home_rooms': advert.home_rooms,
                'home_floor': advert.home_floor,
                'floor_count': advert.floor_count
            })
        return self.Property(**property_args)

    def save(self):
        session = Session()
        # INFO - Ident: Alvito k.,Ežero g., Įrengtas, 94.35m2 18.0a
        # Property = create_property_model(self.table_name)

        for ident, advert in self.places.items():
            existing_house = False
            existing_house = session.query(self.Property).filter_by(url=advert.url).order_by(self.Property.time.desc()).all()

            if existing_house:
                latest_property = existing_house[0]

                has_changes = False

                if latest_property.price != advert.price:
                    has_changes = True

                for property in existing_house:
                    property.last_found_date = datetime.datetime.now()

                if has_changes:
                    log.info("Detected property that is changed: %s", ident)
                    changed_house = self.return_property(advert)
                    session.add(changed_house)
            else:
                log.info("Detected new property. Adding it to database: %s", ident)
                new_house = self.return_property(advert)
                session.add(new_house)

        try:
            session.commit()
            update_last_checked_file(self.table_name)
            print(f"Changes committed to {self.table_name} and tracking file updated.")
        except SQLAlchemyError as e:
            session.rollback()
            print(f"Failed to commit changes to {self.table_name}: {e}")
        session.close()
    
    
    def mark(self, time):
        self.time = time

    def has(self, record):
        ident = str(record)
        if ident not in self.places:
            return False
        place = self.places[ident]
        if not place.adverts:
            return False
        advert = place.adverts[-1]
        return not advert.gone and \
                advert == record

    def define(self, record):
        ident = str(record)
        #TODO In future, when we will read DB data firstly
        #if ident not in self.places:
        self.places[ident] = record
        log.info("Found property '%s'", ident)
        return True

    