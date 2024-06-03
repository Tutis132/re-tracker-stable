# re-tracker

This tool is designed to collect information from www.aruodas.lt page.
You will be able to create your custom real estate profiles, which will periodically search for advertisements in that URL.

In order to use this tool, you need chromium drivers and chrome browser.

Tool is running on Ubuntu 22.04 LTS version VirtualBox image. On different platforms it might not work right now.
All steps and descriptions are written for this specific architecture.

## Installation

### Initial installs

`python3 -m venv venv`\
`source venv/bin/activate`\
`pip3 install -r requirements.txt`\
`python3 app.py`

### Install tools required to work with PostgreSQL
`pip3 install Flask-SQLAlchemy psycopg2-binary`

For more significant applications or those moving to production, it's recommended to handle database initialization and subsequent migrations (adjustments to the database structure as your models change over time) using Flask-Migrate. Flask-Migrate is an extension that handles SQLAlchemy database migrations for Flask applications using Alembic.

`pip3 install Flask-Migrate`


## Install PostgreSQL database

`sudo apt-get update`\
`sudo apt-get install postgresql postgresql-contrib`

Switch to the postgres User:
PostgreSQL creates a user named postgres by default for handling database operations. Switch to this user using:

`sudo -i -u postgres`

Access the PostgreSQL Prompt:
Once switched to the postgres user, access the PostgreSQL interactive prompt by running:

`psql`

Create a Database:
At the PostgreSQL prompt, create a new database. Replace mydatabase with the name you wish to give your database.

`CREATE DATABASE re_tracker_db;`

Create a User:
Still at the PostgreSQL prompt, create a new user. Replace username with your desired username and password with a secure password.

`CREATE USER tester WITH PASSWORD 'tester';`

Grant Privileges:
Grant all privileges on your new database to your new user.

`GRANT ALL PRIVILEGES ON DATABASE re_tracker_db TO tester;`

Exit:
Exit the PostgreSQL prompt with:

`\q`

Return to Your Regular User:
If you switched to the postgres user with `sudo -i -u postgres`, return to your regular user by typing `exit` and pressing Enter.

### Configure PostgreSQL to Allow Remote Connections (Optional)
By default, PostgreSQL is configured to only allow connections from the local machine. If you need to allow remote connections:

Edit the `postgresql.conf` File:
Open the `postgresql.conf` file located in the PostgreSQL directory within `/etc/postgresql/XX/main/`, where XX is the version number of PostgreSQL. You can use nano or any text editor you prefer:

`sudo nano /etc/postgresql/12/main/postgresql.conf`
Find the line `#listen_addresses = 'localhost'` and change it to `listen_addresses = '*'` to allow PostgreSQL to listen for connections on all network interfaces.

Edit the `pg_hba.conf` file, also located in the same directory:

`sudo nano /etc/postgresql/12/main/pg_hba.conf`
Add the following line to the end of the file to allow connections from all IP addresses with password authentication:

```
host    all             all             0.0.0.0/0               md5
Replace 0.0.0.0/0 with a specific IP address or subnet if you want to restrict access.
```

Restart PostgreSQL:
Apply the changes by restarting PostgreSQL:

`sudo systemctl restart postgresql`


### Connect to postgresql database with tester
`psql -U tester -d re_tracker_db -W`
or
`psql -U tester -d re_tracker_db -h localhost -W` password is tester.

List databases
`\l`

Connect to database

`\c re_tracker_db`

List all tables

`\dt`

Select all info from single table

`SELECT * FROM contact_submissions;`

## Connect to GUI

`ssh -L 5000:localhost:5000 name@localhost -p PORT`