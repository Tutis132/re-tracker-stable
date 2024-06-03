# Required for user registration and login
from werkzeug.security import generate_password_hash, check_password_hash
from sqlalchemy import Column, Integer, String, Float, DateTime
from flask_login import UserMixin

import datetime

from app_setup import db

class ContactSubmission(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(120), nullable=False)
    message = db.Column(db.Text, nullable=False)

    def __repr__(self):
        return f'<ContactSubmission {self.name}>'


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(100), unique=True, nullable=False)
    email = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(128))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

# Each type of estate will have its own entry in database.
# Each found specific estate should have entry in database. Inside that entry, there will be timestamps of checks.
# Inside these timestamps there will be information about that estate.

from datetime import datetime
from app_setup import db

def create_property_model(table_name, type='house'):
    if type == 'house':
        class House(db.Model):
            __tablename__ = table_name
            __table_args__ = {'extend_existing': True}

            id = db.Column(db.Integer, primary_key=True)
            time = db.Column(db.DateTime, default=db.func.current_timestamp())
            url = db.Column(db.String)
            address = db.Column(db.String)
            promo = db.Column(db.Float)
            price = db.Column(db.Float)
            square = db.Column(db.Float)
            home_size = db.Column(db.Float)
            site_size = db.Column(db.Float)
            state = db.Column(db.String)
            last_found_date = db.Column(db.DateTime, default=datetime.now())

            def __repr__(self):
                return f'<House {self.address}>'

        return House

    elif type == 'flat':
        class Flat(db.Model):
            __tablename__ = table_name
            __table_args__ = {'extend_existing': True}

            id = db.Column(db.Integer, primary_key=True)
            time = db.Column(db.DateTime, default=db.func.current_timestamp())
            url = db.Column(db.String)
            address = db.Column(db.String)
            promo = db.Column(db.Float)
            price = db.Column(db.Float)
            square = db.Column(db.Float)
            home_size = db.Column(db.Float)
            home_rooms = db.Column(db.Integer)
            home_floor = db.Column(db.Integer)
            floor_count = db.Column(db.Integer)
            last_found_date = db.Column(db.DateTime, default=datetime.now())

            def __repr__(self):
                return f'<Flat {self.address}>'

        return Flat

    else:
        raise ValueError("Unsupported property type")


'''
class House(db.Model):
    __tablename__ = 'houses'

    id = db.Column(db.Integer, primary_key=True)
    time = db.Column(db.DateTime, default=datetime.datetime.utcnow)
    url = db.Column(db.String, unique=True)
    banner = db.Column(db.String)
    address = db.Column(db.String)
    promo = db.Column(db.Float)
    price = db.Column(db.Float)
    square = db.Column(db.Float)
    home_size = db.Column(db.Float)
    site_size = db.Column(db.Float)
    state = db.Column(db.String)
    banner_file = db.Column(db.String)
'''