import os
import pandas as pd
from flask import Flask
from flask import request
from flask import jsonify
from flask import abort

app = Flask(__name__)


@app.route('/estimates', methods=['GET'])
def generate_tasks():
    content = request.get_json()
    #job_id = int(content['jobId'])
    #stop_score = content['stopScore']
    #out_threshold = content['outThreshold']
    #filters_data = content['criteria']
    # if fib.assign_filters() == "filters_assigned":
    #    response = {"message": "filters_assigned"}
    #    return jsonify(response)
    # else:
    #    abort(500, {"message": "error"})
    response = {"message": "estimates"}
    return jsonify(response)
