import configparser
from app_setup import scheduler

import subprocess
from flask import redirect, url_for, request
from threading import Thread
from app_setup import app, scheduler
import sys

def get_weekday_abbr(weekday_name):
    day_mapping = {
        'sunday': 'sun',
        'monday': 'mon',
        'tuesday': 'tue',
        'wednesday': 'wed',
        'thursday': 'thu',
        'friday': 'fri',
        'saturday': 'sat'
    }
    weekday_name = weekday_name.strip().lower()
    return day_mapping.get(weekday_name, None)


def scrape(section, section_type):

    print(f'Section type scrape: {section_type}')

    if section_type == 'house':
        from houses import run_house_scrape
        thread = Thread(target=run_house_scrape, args=(section,))
    elif section_type == 'flat':
        from flats import run_flat_scrape
        thread = Thread(target=run_flat_scrape, args=(section,))
    else:
        return "Error: Unknown type", 400

    thread.start()

def report(section, section_type):

    print(f'Section type report: {section_type}')
    command = ["python3", "./digest.py", "-i", section]
    subprocess.run(command, check=True)

def schedule_job(func, section, type_or_time, frequency, day=None, time="00:00"):
    """
    Schedules a job based on the frequency type.
    func: Function to be scheduled.
    section: Section name from the config.
    type_or_time: Either the property type for scraping or time string for reporting.
    frequency: Frequency type ('daily', 'weekly', 'monthly').
    day: Day of the week or month depending on frequency.
    time: Time of the day in HH:MM format.
    """

    hour, minute = map(int, time.split(':'))
    if frequency == 'monthly':
        scheduler.add_job(func=func, trigger='cron', day=day, hour=hour, minute=minute, args=[section, type_or_time], id=f'{func.__name__}_{section}')
    elif frequency == 'weekly':
        day_abbr = get_weekday_abbr(day) if day else None
        scheduler.add_job(func=func, trigger='cron', day_of_week=day_abbr, hour=hour, minute=minute, args=[section, type_or_time], id=f'{func.__name__}_{section}')
    elif frequency == 'daily':
        scheduler.add_job(func=func, trigger='cron', hour=hour, minute=minute, args=[section, type_or_time], id=f'{func.__name__}_{section}')

def print_scheduled_jobs():
    jobs = scheduler.get_jobs()
    print("Scheduled Jobs:")
    for job in jobs:
        print(f"Job ID: {job.id}")
        print(f"Function: {job.func_ref}")
        print(f"Next Run: {job.next_run_time}")
        print(f"Trigger: {job.trigger}")
        print()

def schedule_jobs_from_config():
    config = configparser.ConfigParser()
    config.read('estate.conf')

    scheduler.remove_all_jobs()

    for section in config.sections():
        print(f'Section: {section}')

        type = config.get(section, 'type', fallback="")

        scrape_frequency = config.get(section, 'scrapingfrequency', fallback="")
        scrape_day = config.get(section, 'scrapingday', fallback="")
        scrape_time = config.get(section, 'scrapingtime', fallback="00:00")

        report_frequency = config.get(section, 'reportfrequency', fallback="")
        report_day = config.get(section, 'reportday', fallback="")
        report_time = config.get(section, 'reporttime', fallback="00:00")

        if scrape_frequency:
            print("Scrape schedule")
            schedule_job(scrape, section, type, scrape_frequency, day=scrape_day, time=scrape_time)
        if report_frequency:
            print("Report schedule")
            schedule_job(report, section, section, report_frequency, day=report_day, time=report_time)
