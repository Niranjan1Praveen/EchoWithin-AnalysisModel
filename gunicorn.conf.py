# gunicorn.conf.py
workers = 1
threads = 2
timeout = 120
max_requests = 10
max_requests_jitter = 5
preload_app = True
worker_class = 'sync'