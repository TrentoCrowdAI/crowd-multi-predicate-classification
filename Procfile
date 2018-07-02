web: gunicorn flask_app:app
worker: celery -E -A flask_app.celery worker