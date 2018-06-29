import os
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort
import uuid
import time
from flask_rq2 import RQ
import redis
import json

from estimator import Estimator

app = Flask(__name__)
rq = RQ(app, default_timeout=18000)
r = redis.StrictRedis.from_url(
    app.config['RQ_REDIS_URL'], decode_responses=True)


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
    filters_difficulty = content['filtersDifficulty']
    stop_score = content.get('stopScore', 100)
    iterations = content.get('iterations', 50)
    single_run = content.get('single', False)
    params = {
        'filters_num': filters_num,
        'items_num': items_num,
        'baseround_items': baseround_items,
        'items_per_worker': items_per_worker,
        'votes_per_item': votes_per_item,
        'filters_select': filters_selectivity,
        'filters_dif': filters_difficulty,
        'worker_tests': initial_tests,
        'stop_score': stop_score,
        'iter_num': iterations,
        'lr': 5,
        'z': 0.3,
        'theta': 0.3
    }
    token = str(uuid.uuid4())
    __run.queue(params, single_run, token)
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
        'status': r.get(f"{token}_status")
    }

    if status == None:
        payload['status'] = 'NONE'
    return jsonify(payload)


@rq.job
def __run(params, single_run, token):
    print(f"Running {token}")
    start_time = time.time()
    r.set(f"{token}_status", 'IN_PROGRESS')
    estimator = Estimator(params)
    output = estimator.run(single_run)
    r.set(token, output.to_json(orient='records'))
    r.set(f"{token}_status", 'DONE')
    total_time = time.time() - start_time
    print(f"{token} DONE. Time: {total_time} seconds")
