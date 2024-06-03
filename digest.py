#!/usr/bin/env python

import os
import json
import datetime
import configparser
import argparse
import logging
import logging.config
import statistics

from db_setup import Session

logging.config.fileConfig('logging.ini')

log = logging.getLogger('estate.digest')

curdir = os.path.dirname(__file__)


def humanise(value):
    if not value:
        return str(0)
    value = value / 1000
    if abs(value) < 1000:
        return f"{value:.2f}k"
    value = value / 1000
    if abs(value) < 1000:
        return f"{value:.2f}M"
    value = value / 1000
    if abs(value) < 1000:
        return f"{value:.2f}B"
    return str(value)


def dtval(tstr):
    return datetime.datetime.strptime(tstr, "%Y-%m-%d %H:%M:%S")

class Table(dict):

    def __init__(self, init=None, safe=True):
        if not init:
            init = {}
        super().__init__(init)
        self.safe = safe

    def __getattr__(self, attr):
        if attr not in self:
            if not self.safe:
                raise AttributeError
            else:
                return None
        return self[attr]

    def __setattr__(self, attr, data):
        if attr == 'safe':
            self.__dict__[attr] = data
            return

        self[attr] = data
        if data is None:
            del self[attr]


class PropertyRecord(Table):

    def __init__(self, file_name):
        super().__init__(self)
        self.datetime = None
        with open(file_name, "r", encoding="utf-8") as f:
            data = json.load(f)
            self.update(data)

    def __str__(self):
        rsv = '*' if self.reserved else ''
        return f"{rsv}{self.address}"

    def dt(self):
        if not self.datetime:
            self.datetime = dtval(self.time)
        return self.datetime

    def place_summary(self):
        return f"{self.address}"

    def price_summary(self):
        return f"{self.price_total()} | {self.price_square()} | {self.price_promo()}"

    def price_total(self):
        return f"{self.price}Eur"

    def price_square(self):
        return f"{self.square}Eur/sq"

    def price_promo(self):
        return f"{self.promo}%" if self.promo else "n/a"


class SiteRecord(PropertyRecord):
    """
        self.time = spot['time']
        self.url = spot['url']
        self.banner = spot['banner']
        self.address = spot['address']
        self.promo = spot['promo']
        self.price = spot['price']
        self.square = spot['square']
        self.site_size = spot['site_size']
        self.site_intent = spot['site_intent']
    """

    def __str__(self):
        rsv = '*' if self.reserved else ''
        return f"{rsv}{self.address}, {self.site_size}a, {self.site_intent}"

    def place_summary(self):
        return f"{self.address} | {self.site_size}a | {self.site_intent}"

    def price_square(self):
        return f"{self.square}Eur/a"


class FlatRecord(PropertyRecord):
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
    """

    def __str__(self):
        rsv = '*' if self.reserved else ''
        return f"{rsv}{self.address}, {self.home_size}m2, {self.home_rooms}k, {self.home_floor}/{self.floor_count}a"

    def place_summary(self):
        return f"{self.address} | {self.home_size}m2 | {self.home_rooms}k | {self.home_floor}/{self.floor_count}a"

    def price_square(self):
        return f"{self.square}Eur/m2"

def create_house_model(table_name='houses'):
    from app_setup import db
    class House(db.Model):
        __tablename__ = table_name
        __table_args__ = {'extend_existing': True}

        id = db.Column(db.Integer, primary_key=True)
        time = db.Column(db.DateTime, default=db.func.current_timestamp())
        url = db.Column(db.String, unique=False)
        address = db.Column(db.String)
        promo = db.Column(db.Float)
        price = db.Column(db.Float)
        square = db.Column(db.Float)
        home_size = db.Column(db.Float)
        site_size = db.Column(db.Float)
        state = db.Column(db.String)
        last_found_date = db.Column(db.DateTime, default=datetime.datetime.now())
    
    return House

def create_flat_model(table_name='flat'):
    from app_setup import db

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
        last_found_date = db.Column(db.DateTime, default=datetime.datetime.now())

    return Flat

class HouseRecord(PropertyRecord):
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
    """

    def __str__(self):
        rsv = '*' if self.reserved else ''
        return f"{rsv}{self.address}, {self.home_size}m2, {self.site_size}a, {self.state}"

    def place_summary(self):
        return f"{self.address} | {self.home_size}m2 | {self.site_size}a | {self.state}"

    def price_square(self):
        return f"{self.square}Eur/m2"


class Message:

    def __init__(self):
        self.blocks = []

    def __str__(self):
        lines = [" | ".join(parts) for parts in self.blocks]
        return "\n".join(lines)

    def add_line(self, *parts):
        self.blocks.append(list(parts))

    def add_part(self, *parts):
        if len(self.blocks) == 0:
            self.add_line(*parts)
            return

        self.blocks[-1].extend(list(parts))

def create_directory(dir_name, secondary_dir = False):
    path = dir_name
    if secondary_dir:
        path = os.path.join(dir_name, secondary_dir)

    if not os.path.exists(path):
        os.makedirs(path)
    return path

def append_message_to_profile_report(section, message):

    profile_name = section.name
    reports_dir = 'reports'

    create_directory(reports_dir)

    file_path = os.path.join(reports_dir, f"{profile_name}.txt")
    message_str = str(message)

    with open(file_path, "a") as file:
        file.write(message_str + "\n\n")

def str_to_datetime(date_str):
    converted = datetime.datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%S.%f")
    print(converted)

    return converted

def is_place_new(place, span):
    return place.time >= datetime.datetime.now() - span

def is_place_gone(place, span, last_checked):
    if place.last_found_date is None:
        return True

    return place.last_found_date < last_checked and place.last_found_date >= last_checked - span

def is_place_back(place, span, last_checked):
    return place.gone_since < last_checked and place.time >= last_checked - span

def get_last_checked_date(table_name):
    filename = 'last_checked.json'
    try:
        with open(filename, 'r') as file:
            last_checked = json.load(file)
            return last_checked.get(table_name)
    except FileNotFoundError:
        return None

def summary_place_counts(msg, span, places, category, last_checked):
    places_total = 0
    places_new = 0
    places_gone = 0
    places_back = 0
    places_sweep = 0

    last_checked_datetime = str_to_datetime(last_checked)

    for place in places:
        pgone = False

        if is_place_new(place, span):
            places_new += 1
        if is_place_gone(place, span, last_checked_datetime):
            # TODO update property status when it is gone and later exclude from calculations
            places_gone += 1
            pgone = True
        #if place.is_back(span): TODO perhaps later
        #    places_back += 1
        #if pnew and pgone:
        #    places_sweep += 1
        if not pgone:
            places_total += 1

    msg.add_line(f"{category} count")
    msg.add_part(f"total {places_total}")
    if places_new:
        msg.add_part(f"{places_new} new")
    if places_gone:
        msg.add_part(f"{places_gone} gone")
    if not places_new and not places_gone:
        msg.add_part("unchanged")

def summary_place_market(msg, span, places, category, last_checked):
    price_inc, price_dec, price_stay, price_all = [], [], [], []
    total_all_a, total_all_b, total_inc_a, total_inc_b = 0, 0, 0, 0
    total_dec_a, total_dec_b, total_stay_a, total_stay_b = 0, 0, 0, 0
    count_all_a, count_all_b = 0, 0

    last_checked_datetime = str_to_datetime(last_checked)
    end_date = datetime.datetime.now()
    start_date = end_date + span

    from collections import defaultdict
    grouped_by_url = defaultdict(list)
    for place in places:
        if start_date <= place.time <= end_date:
            grouped_by_url[place.url].append(place)
    

    for url, property_records in grouped_by_url.items():
        sorted_records = sorted(property_records, key=lambda x: x.time)
        
        # Leave it for later debug
        #for place in sorted_records:
        #    print(f"    Time: {place.time}, Price: {place.price}")

        prices = [record.price for record in sorted_records]
        if len(prices) > 1:
            print("Current and previous price found")
            current_price = prices[-1]
            previous_price = prices[0]
            print(f"current price {current_price}")
            print(f"previous price {previous_price}")
            total_all_a += current_price
            count_all_a += 1
            total_all_b += previous_price
            count_all_b += 1
        else:
            print("Only current price found")
            current_price = prices[0]
            previous_price = current_price
            total_all_a += current_price
            count_all_a += 1
        
        delta = current_price - previous_price
        print(f"[{category}] Change: {current_price:10.1f} -> {previous_price:10.1f} = {delta:10.1f} - {sorted_records[0].address}")
        if delta > 0:
            price_inc.append(delta)
            total_inc_a += current_price
            total_inc_b += previous_price
        elif delta < 0:
            price_dec.append(delta)
            total_dec_a += current_price
            total_dec_b += previous_price
        else:
            if not is_place_gone(sorted_records[-1], span, last_checked_datetime):
                price_stay.append(delta)
        price_all.append(delta)
        total_stay_a += current_price
        total_stay_b += previous_price

    total_inc_c = total_inc_a - total_inc_b
    if total_inc_b:
        total_inc_pct = (total_inc_c / total_inc_b) * 100
    else:
        total_inc_pct = 0

    print(f"[{category}] Increase: {total_inc_b} => {total_inc_a} / {total_inc_c} / {total_inc_pct}")

    total_dec_c = total_dec_a - total_dec_b
    if total_dec_b:
        total_dec_pct = (total_dec_c / total_dec_b) * 100
    else:
        total_dec_pct = 0

    print(f"[{category}] Decrease: {total_dec_b} => {total_dec_a} / {total_dec_c} / {total_dec_pct}")

    total_all_c = total_all_a - total_all_b
    if total_all_b:
        total_all_pct = (total_all_c / total_all_b) * 100
    else:
        total_all_pct = 0

    print(f"[{category}] Total lots: {total_all_b} => {total_all_a} / {total_all_c} / {total_all_pct}")

    total_stay_c = total_stay_a - total_stay_b
    if total_stay_b:
        total_stay_pct = (total_stay_c / total_stay_b) * 100
    else:
        total_stay_pct = 0

    print(f"[{category}] Stayed lots: {total_stay_b} => {total_stay_a} / {total_stay_c} / {total_stay_pct}")

    # Apply min-max filtering using average standard deviation
    try:
        stdev = statistics.pstdev(price_all)
        mean = statistics.mean(price_all)
        th_up = mean + stdev
        th_dn = mean - stdev
    except:
        th_up = 0
        th_dn = 0

    try:
        list_inc = [e for e in price_inc if e <= th_up and e >= th_dn]
        avg_inc = statistics.mean(list_inc)
        sum_inc = sum(list_inc)
    except:
        if price_inc:
            avg_inc = statistics.mean(price_inc)
            sum_inc = sum(price_inc)
        else:
            avg_inc = 0
            sum_inc = 0

    print(f"[{category}] Consolidated increase: avg {avg_inc}, sum {sum_inc}")

    try:
        list_dec = [e for e in price_dec if e <= th_up and e >= th_dn]
        avg_dec = statistics.mean(list_dec)
        sum_dec = sum(list_dec)
    except:
        if price_dec:
            avg_dec = statistics.mean(price_dec)
            sum_dec = sum(price_dec)
        else:
            avg_dec = 0
            sum_dec = 0

    print(f"[{category}] Consolidated decrease: avg {avg_dec}, sum {sum_dec}")

    msg.add_line(f"{category} price")
    msg.add_part(f"E {humanise(total_all_a)} Eur")
    if total_stay_c:
        msg.add_part(f"Δ {humanise(total_stay_c)} Eur")
        msg.add_part(f"{total_stay_pct:.2f} %")
    else:
        msg.add_part("stable")

    if price_inc:
        msg.add_line(f"{category} increased {len(price_inc)}")
        msg.add_part(f"E {humanise(sum_inc)} Eur")
        msg.add_part(f"μ {humanise(avg_inc)} Eur")
        msg.add_part(f"{total_inc_pct:.2f} %")
    if price_dec:
        msg.add_line(f"{category} decreased {len(price_dec)}")
        msg.add_part(f"E {humanise(sum_dec)} Eur")
        msg.add_part(f"μ {humanise(avg_dec)} Eur")
        msg.add_part(f"{total_dec_pct:.2f} %")
    if price_stay:
        msg.add_line(f"{category} stable {len(price_stay)}")

def return_property_model(type, table):
    if type == 'house':
        Property = create_house_model(table)
    elif type == 'flat':
        Property = create_flat_model(table)

    return Property

def read_data(type, table):
    session = Session()
    # Integrate this function to work according to category (house, flat)
    model = return_property_model(type, table)
    places = session.query(model).all()
    session.close()

    return places

def summarize_properties(section, frequency):
    dnow = datetime.datetime.now()

    if frequency == 'monthly':
        dold = dnow.replace(month=dnow.month - 1)
        span = dold - dnow
        d = dold.strftime("%Y-%m")
        summary = f"{dnow}. Monthly summary: {d}"
    elif frequency == 'weekly':
        span = datetime.timedelta(weeks=-1)
        df = (datetime.datetime.now() + span).strftime("%Y-%m-%d")
        dt = datetime.datetime.now().strftime("%Y-%m-%d")
        summary = f"{dnow}. Weekly summary: {df} ~ {dt}"
    elif frequency == 'daily':
        span = datetime.timedelta(days=-1)
        d = (datetime.datetime.now() + span).strftime("%Y-%m-%d")
        summary = f"{dnow}. Daily summary: {d}"
    else:
        return

    log.info("Summarizing places since %s",
             (datetime.datetime.now() + span).strftime("%Y-%m-%d"))

    queue_msg = {}
    
    queue = section['queue']
    table = section['table']
    type = section['type']

    last_checked = get_last_checked_date(table)

    if queue not in queue_msg:
        msg = Message()
        msg.add_line(f"## {summary}")
        msg.add_line("────────────────────────────────────────")
        queue_msg[queue] = msg
    else:
        msg = queue_msg[queue]

    
    places = read_data(type, table)

    # Required for filtering same properties with different prices
    unique_places = {}
    for place in places:
        if place.url not in unique_places:
            unique_places[place.url] = place

    filtered_places_for_counts = list(unique_places.values())

    summary_place_counts(msg, span, filtered_places_for_counts, type, last_checked)
    summary_place_market(msg, span, places, type, last_checked)
    msg.add_line("────────────────────────────────────────")

    append_message_to_profile_report(section, msg)

def calculate_daily_average_price_per_square_meter(type, table):
    session = Session()
    Property = return_property_model(type, table)

    oldest_property = session.query(Property).order_by(Property.time).first()
    oldest_time = oldest_property.time
    oldest_time_converted = oldest_time.strftime("%Y-%m-%d")
    print(f'Oldest property time {oldest_time_converted}')
   
    current_date = datetime.datetime.strptime(oldest_time_converted, "%Y-%m-%d").date()
    end_date = datetime.datetime.now().date()

    daily_averages = {}

    while current_date <= end_date:
        properties = session.query(Property).filter(
            Property.time <= current_date,
            Property.last_found_date >= current_date
        ).all()

        total_square_price = sum([prop.square for prop in properties])
        num_properties = len(properties)

        average_price_per_sqm = total_square_price / num_properties if num_properties > 0 else 0

        print(f"Day {current_date.strftime('%Y-%m-%d')}: Average price per square meter: {average_price_per_sqm}")

        if average_price_per_sqm != 0:
            daily_averages[current_date.strftime('%Y-%m-%d')] = round(average_price_per_sqm)

        current_date += datetime.timedelta(days=1)

    session.close()
    return daily_averages


def predict_price_arima(section, data):

    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt

    df = pd.DataFrame(list(data.items()), columns=['date', 'average_price'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    model = ARIMA(df['average_price'], order=(1, 1, 1)) 
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=30)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['average_price'], label='Observed')
    plt.plot(mean_forecast.index, mean_forecast.values, color='red', label='Forecast')
    plt.fill_between(mean_forecast.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1], color='pink')
    plt.title('Average Price Forecast')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()

    base_dir = os.path.join('static', 'predictions')
    #base_dir = 'predictions'
    path = create_directory(base_dir, section)
    print(path)

    plot_filename = f"arima_forecast_{datetime.datetime.now().date().strftime('%Y-%m-%d')}.png"

    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')
    plt.close()

def predict_price_sarima(section, data):
    
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import matplotlib.pyplot as plt

    df = pd.DataFrame(list(data.items()), columns=['date', 'average_price'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    model = SARIMAX(df['average_price'], order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=30)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df['average_price'], label='Observed')
    plt.plot(mean_forecast.index, mean_forecast, color='red', label='Forecast')
    plt.fill_between(mean_forecast.index,
                    confidence_intervals.iloc[:, 0],
                    confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Average Price Forecast with SARIMA (Weekly Seasonality)')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()
    
    base_dir = os.path.join('static', 'predictions')
    #base_dir = "predictions"
    path = create_directory(base_dir, section)
    print(path)

    plot_filename = f"sarima_forecast_{datetime.datetime.now().date().strftime('%Y-%m-%d')}.png"

    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')
    plt.close()

# This is experimental to forecast and compare future
def arima_test(section, data):
    import pandas as pd
    from statsmodels.tsa.arima.model import ARIMA
    import matplotlib.pyplot as plt
    import os
    import numpy as np

    df = pd.DataFrame(list(data.items()), columns=['date', 'average_price'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    train = df.iloc[:130]
    test = df.iloc[130:]

    model = ARIMA(train['average_price'], order=(2, 2, 2))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=68)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    mre = np.mean(np.abs((test['average_price'] - mean_forecast) / test['average_price']))
    print(f"Mean Relative Error: {mre * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(df['average_price'], label='Observed (Complete Data)', color='blue')  # Full dataset

    combined_index = train.index.union(mean_forecast.index)
    combined_data = pd.concat([train['average_price'], mean_forecast])

    plt.plot(combined_index, combined_data, color='red', label='Forecast')
    plt.fill_between(mean_forecast.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Average Price Forecast with ARIMA')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()

    base_dir = os.path.join('static', 'predictions')
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    path = os.path.join(base_dir, section)
    if not os.path.exists(path):
        os.makedirs(path)

    plot_filename = f"arima_forecast_222_130days_68days_data_test.png"
    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')
    plt.close()
    print(f"Plot saved to {plot_path}")


def sarima_test(section, data):
    
    import pandas as pd
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    import matplotlib.pyplot as plt
    import numpy as np

    df = pd.DataFrame(list(data.items()), columns=['date', 'average_price'])
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    train = df.iloc[:130]
    test = df.iloc[130:]

    model = SARIMAX(train['average_price'], order=(2, 2, 2), seasonal_order=(2, 2, 2, 7))
    fitted_model = model.fit()

    print(fitted_model.summary())

    forecast = fitted_model.get_forecast(steps=68)
    mean_forecast = forecast.predicted_mean
    confidence_intervals = forecast.conf_int()

    mre = np.mean(np.abs((test['average_price'] - mean_forecast) / test['average_price']))
    print(f"Mean Relative Error: {mre * 100:.2f}%")

    plt.figure(figsize=(10, 5))
    plt.plot(df['average_price'], label='Observed (Complete Data)', color='blue')  # Full dataset

    combined_index = train.index.union(mean_forecast.index)
    combined_data = pd.concat([train['average_price'], mean_forecast])

    plt.plot(combined_index, combined_data, color='red', label='Forecast')
    plt.fill_between(mean_forecast.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1], color='pink', alpha=0.3)
    plt.title('Average Price Forecast with SARIMA (Weekly Seasonality)')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.legend()
    
    base_dir = os.path.join('static', 'predictions')
    path = create_directory(base_dir, section)
    print(path)

    plot_filename = f"sarima_forecast_222222_130days_68days_data_test.png"

    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')
    plt.close()

# This function is only used for experiments.
def average_price_graph(section, daily_averages):
    import pandas as pd

    dates = list(daily_averages.keys())
    prices = list(daily_averages.values())

    df = pd.DataFrame({
        'Date': pd.to_datetime(dates),
        'Average Price': prices
    })

    #df.to_csv('daily_average_prices.csv', index=False)
    df = pd.read_csv('daily_average_prices.csv')
    df['Date'] = pd.to_datetime(df['Date'])

    df.sort_values('Date', inplace=True)

    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Average Price'], color='red')
    plt.title('Daily Average Price Per Square Meter')
    plt.xlabel('Date')
    plt.ylabel('Average Price per Square Meter')
    plt.grid(True)
    plt.xticks(rotation=45)
    plt.tight_layout()

    base_dir = os.path.join('static', 'predictions')
    path = create_directory(base_dir, section)
    print(path)

    plot_filename = f"test_average_price.png"

    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')
    plt.close()

def create_dataset(X, y, time_steps=1):
    import numpy as np
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X.iloc[i:(i + time_steps)].values
        Xs.append(v)
        ys.append(y.iloc[i + time_steps])
    return np.array(Xs), np.array(ys)

# LSTM does not work on server, where this tool is running, because AVX is not supported.
# Calculations are done on different PC.
# Firstly average prices results were exported to csv file, then this function reads and adjusts.
def lstm_predict(section):

    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.optimizers import Adam

    data = pd.read_csv('daily_average_prices.csv', parse_dates=['Date'], index_col='Date')

    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)

    train_size = int(len(data) * 0.5)
    val_size = int(len(data) * 0.1)
    test_size = len(data) - train_size - val_size

    train_data = scaled_data[:train_size]
    val_data = scaled_data[train_size:train_size + val_size]
    test_data = scaled_data[train_size + val_size:]

    time_steps = 5
    units_count = 50
    batch_size = 128
    epochs = 50
    learning_rate = 0.001
    loss_function = 'mean_absolute_error'

    def create_sequences(data, time_steps):
        X, y = [], []
        for i in range(time_steps, len(data)):
            X.append(data[i-time_steps:i])
            y.append(data[i])
        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_data, time_steps)
    X_val, y_val = create_sequences(val_data, time_steps)
    X_test, y_test = create_sequences(test_data, time_steps)

    model = Sequential()
    model.add(LSTM(units_count, input_shape=(time_steps, 1), activation='tanh', kernel_regularizer=l2(0.01)))
    model.add(Dense(1))

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss=loss_function)

    #model.compile(optimizer='adam', loss=loss_function)

    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val), verbose=1)

    predictions = model.predict(X_test)
    predictions_inverse = scaler.inverse_transform(predictions)

    y_test_inverse = scaler.inverse_transform(y_test)

    mae = mean_absolute_error(y_test_inverse, predictions_inverse)
    rmse = np.sqrt(mean_squared_error(y_test_inverse, predictions_inverse))
    mre = np.mean(np.abs((y_test_inverse - predictions_inverse) / y_test_inverse))

    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'MRE: {mre}')

    full_predictions = np.full(shape=(len(data),), fill_value=np.nan)
    full_predictions[train_size + val_size + time_steps:] = predictions_inverse.flatten() 
    
    plt.figure(figsize=(10, 6))
    plt.plot(data.index, scaler.inverse_transform(scaled_data), label='Real Average Price')
    plt.plot(data.index, full_predictions, label='Predicted Average Price', color='orange')
    plt.title('Real Estate Price Prediction with LSTM.')
    plt.xlabel('Time')
    plt.ylabel('Average Price')
    plt.legend()

    base_dir = os.path.join('static', 'predictions')
    path = create_directory(base_dir, section)
    print(path)

    plot_filename = f"arima_forecast_{datetime.datetime.now().date().strftime('%Y-%m-%d')}.png"

    plot_path = os.path.join(path, plot_filename)
    plt.savefig(plot_path, format='png')

    plt.close()

def run_report():
    # TODO When triggering report manually need to select daily/weekly/monthly report.
    config = configparser.ConfigParser()
    config.read(f'{curdir}/estate.conf')

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', action='store', dest='input')
    args = parser.parse_args()
    section_name = args.input

    section = config[section_name]
    table = section['table']
    type = section['type']

    daily_averages = calculate_daily_average_price_per_square_meter(type, table)

    # Code below is required only for custom predictions of already known values.
    #keys_to_delete = list(daily_averages.keys())[130:]
    #for key in keys_to_delete:
    #    del daily_averages[key]
    #print(daily_averages)

    # average_price_graph(section_name, daily_averages)
    predict_price_arima(section_name, daily_averages)
    predict_price_sarima(section_name, daily_averages)
    # This is commented and experimented only on different server.
    # lstm_predict(section)

    # Function calls below are for experiments.
    # sarima_test(section_name, daily_averages)
    # arima_test(section_name, daily_averages)

    if section and section['reportfrequency']:
        summarize_properties(section, section['reportfrequency'])
        pass
    else:
        print(f"Section not found: {args.input}")

if __name__ == "__main__":
    run_report()
