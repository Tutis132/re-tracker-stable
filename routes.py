
import sys
import os

from flask import render_template, request, redirect, url_for, flash
from app_setup import app, db
from models import ContactSubmission

# Required for user login and registration
from flask_login import login_user, current_user, logout_user, login_required
from forms import RegistrationForm, LoginForm
from models import User

from config_handler import load_profiles, save_profile, delete_profile, get_profile

from sqlalchemy.exc import ProgrammingError

from models import create_property_model

from urllib.parse import urlparse, urljoin
from flask import request, url_for, redirect

from threading import Thread

@app.route('/')
def landing_page():
    return render_template('landing_page.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'GET':
        return render_template('contact.html')

@app.route('/market_insights')
@login_required
def market_insights():

    all_profiles = load_profiles()
    profiles_dir = 'reports/'

    profiles = [profile for profile in all_profiles if profile.lower() != 'default']

    latest_reports = {}

    for profile in profiles:
        file_path = os.path.join(profiles_dir, f"{profile}.txt")
        try:
            with open(file_path, "r") as file:
                content = file.read().strip()
                reports = [report for report in content.split('##') if report.strip()]
                if reports:
                    latest_reports[profile] = '##' + reports[-1].strip()
        except FileNotFoundError:
            latest_reports[profile] = "No report available for this profile."

    return render_template('market_insights.html', latest_reports=latest_reports, profiles=profiles)

@app.route('/price_predictions')
@login_required
def price_predictions():
    import datetime
    profiles = load_profiles()

    predictions = {}

    for section in profiles:
        predictions_dir_search = f'static/predictions/{section}/'
        #TODO fix
        predictions_dir = f'predictions/{section}/'
        if os.path.exists(predictions_dir_search):
            files = [f for f in os.listdir(predictions_dir_search) if f.startswith('arima_forecast') and f.endswith('.png')]
            files.sort(key=lambda x: datetime.datetime.strptime(x.split('_')[-1].split('.')[0], '%Y-%m-%d'), reverse=True)
            if files:
                predictions[section] = os.path.join(predictions_dir, files[0])
            else:
                predictions[section] = None
        else:
            predictions[section] = None

    return render_template('price_predictions.html', predictions=predictions, profiles=profiles)

@app.route('/investment_opportunities')
@login_required
def investment_opportunities():
    return render_template('investment_opportunities.html')

@app.route('/submit_contact_form', methods=['POST'])
@login_required
def submit_contact_form():
    name = request.form.get('name')
    email = request.form.get('email')
    message = request.form.get('message')
    print(f"Received message from {name} ({email}): {message}")

    submission = ContactSubmission(name=name, email=email, message=message)
    db.session.add(submission)
    db.session.commit()
    flash('Thank you for your message. We will get back to you soon.', 'success')
    return redirect(url_for('contact'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('landing_page'))
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(username=form.username.data, email=form.email.data)
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        flash('Your account has been created! You are now able to log in', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

def is_safe_url(target):
    ref_url = urlparse(request.host_url)
    test_url = urlparse(urljoin(request.host_url, target))
    return test_url.scheme in ('http', 'https') and ref_url.netloc == test_url.netloc

@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('landing_page'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(email=form.email.data).first()
        if user and user.check_password(form.password.data):
            login_user(user)
            flash('You have successfully logged in.', 'success')
            next_page = request.args.get('next')
            if not next_page or not is_safe_url(next_page):
                next_page = url_for('landing_page')
            return redirect(next_page)
        else:
            flash('Login Unsuccessful. Please check email and password', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.after_request
def after_request(response):
    db.session.commit()
    return response

@app.route('/logout')
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('landing_page'))

@app.route('/trigger-scrape', methods=['POST'])
@login_required
def trigger_scrape():
    section = request.form.get('section')
    section_type = request.form.get('type')

    if section_type == 'house':
        from houses import run_house_scrape
        thread = Thread(target=run_house_scrape, args=(section,))
    elif section_type == 'flat':
        from flats import run_flat_scrape
        thread = Thread(target=run_flat_scrape, args=(section,))
    else:
        return "Error: Unknown type", 400

    thread.start()
    return redirect(url_for('automation'))

import subprocess
@app.route('/trigger-report', methods=['POST'])
@login_required
def trigger_report():
    section = request.form.get('section')
    command = ["python3", "./digest.py", "-i", section]
    subprocess.run(command, check=True)
    return redirect(url_for('automation'))

@app.route('/data/<table_name>/<property_type>')
@login_required
def data(table_name, property_type):

    print(property_type)

    Property = create_property_model(table_name, property_type)

    try:
        property = Property.query.all()
    except ProgrammingError:
        flash('No data available yet for the specified profile. Please run the scraping process first.', 'warning')
        property = []
    return render_template('data.html', houses=property, table_name=table_name, property_type=property_type)

@app.route('/automation', methods=['GET', 'POST'])
@login_required
def automation():
    if request.method == 'POST':
        name = request.form['name']
        profile_data = {k: v for k, v in request.form.items() if k != 'name'}
        save_profile(name, **profile_data)
        return redirect(url_for('automation'))

    profiles = load_profiles()
    return render_template('automation.html', profiles=profiles)

@app.route('/delete-profile/<profile_name>', methods=['POST'])
@login_required
def delete_profile_route(profile_name):
    delete_profile(profile_name)
    flash('Profile deleted successfully.')
    return redirect(url_for('automation'))

@app.route('/edit-profile/<profile_name>', methods=['GET', 'POST'])
@login_required
def edit_profile(profile_name):
    if request.method == 'POST':
        profile_data = {k: v for k, v in request.form.items()}
        save_profile(profile_name, **profile_data)
        flash('Profile updated successfully.')
        return redirect(url_for('automation'))

    profile_data = get_profile(profile_name)
    if profile_data is None:
        flash('Profile not found.', 'error')
        return redirect(url_for('automation'))

    return render_template('edit_profile.html', profile_name=profile_name, profile_data=profile_data)

@app.route('/reports/<profile_name>')
@login_required
def reports(profile_name):
    file_path = os.path.join('reports', f"{profile_name}.txt")

    report_content = ""

    try:
        with open(file_path, "r") as file:
            report_content = file.read()
    except FileNotFoundError:
        flash(f"No report available for profile: {profile_name}.", "warning")

    return render_template('reports.html', profile_name=profile_name, report_content=report_content)
