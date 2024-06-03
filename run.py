
from app_setup import app
from app_setup import db

from scheduler import schedule_jobs_from_config, print_scheduled_jobs

if __name__ == '__main__':
    #with app.app_context():
    #    db.create_all()
    
    schedule_jobs_from_config()
    print_scheduled_jobs()


    app.run(debug=False)
