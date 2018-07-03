import os
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
import uuid
import time
import redis
import json

from estimator import Estimator
from tasks import make_celery

app = Flask(__name__)
app.config.update(
    CELERY_BROKER_URL=os.environ.get(
        'REDIS_URL', None) or 'redis://localhost:6379'
)
celery = make_celery(app)
r = redis.StrictRedis.from_url(
    app.config['CELERY_BROKER_URL'], decode_responses=True)


@app.route('/estimates', methods=['POST'])
def estimate():
    content = request.get_json()
    items_per_worker = content['itemsPerWorker']
    votes_per_item = content.get('votesPerItem', 0)
    initial_tests = content.get('initialTests', 0)
    items_num = content['itemsNum']
    filters_num = content['filtersNum']
    baseround_items = content['baseroundItems']
    filters_selectivity = content['filtersSelectivity']
    stop_score = content.get('stopScore', 100)
    iterations = content.get('iterations', 50)
    single_run = content.get('single', False)
    fixed_votes = content.get('fixedVotes', False)
    params = {
        'filters_num': filters_num,
        'items_num': items_num,
        'baseround_items': baseround_items,
        'items_per_worker': items_per_worker,
        'votes_per_item': votes_per_item,
        'filters_select': filters_selectivity,
        'worker_tests': initial_tests,
        'stop_score': stop_score,
        'iter_num': iterations,
        'lr': 5,
        'z': 0.3,
        'theta': 0.3
    }
    token = str(uuid.uuid4())
    r.set(f"{token}_status", 'IN_PROGRESS')
    __run.delay(params, single_run, fixed_votes, token)
    payload = {
        'token': token
    }
    return jsonify(payload)


@app.route('/estimates/<string:token>', methods=['GET'])
def get_estimate(token):
    status = r.get(f"{token}_status")

    if(status == 'DONE'):
        estimate = json.loads(r.get(token))
        return jsonify(estimate)
    else:
        payload = {
            'msg': 'The estimation is still in-progress'
        }
        return jsonify(payload)


@app.route('/status/<string:token>', methods=['GET'])
def get_status(token):
    status = r.get(f"{token}_status")
    payload = {
        'status': status
    }

    if status == None:
        payload['status'] = 'NONE'
    return jsonify(payload)


@celery.task
def __run(params, single_run, fixed_votes, token):
    print(f"Running {token}")
    start_time = time.time()
    estimator = Estimator(params)
    output = estimator.run(single_run, fixed_votes)
    r.set(token, output.to_json(orient='records'))
    r.set(f"{token}_status", 'DONE')
    total_time = time.time() - start_time
    print(f"{token} DONE. Time: {total_time} seconds")
