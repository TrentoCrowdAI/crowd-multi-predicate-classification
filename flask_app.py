import os
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort

from estimator import Estimator

app = Flask(__name__)


@app.route('/estimates', methods=['POST'])
def estimate():
    content = request.get_json()
    items_per_worker = content['itemsPerWorker']
    votes_per_item = content['votesPerItem']
    initial_tests = content['initialTests']
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
    estimator = Estimator(params)
    output = estimator.run(single_run)
    response = app.response_class(
        response=output.to_json(orient='records'),
        status=200,
        mimetype='application/json'
    )
    return response
