from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import join_room

import subprocess
import multitasking
import json
import joblib

from pandas import DataFrame

from src.predict_failing_tests import predict_failing_tests

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

test_data = {}


loaded_object = joblib.load('flask_simple_decision_tree.joblib')
path = "/home/dominik/Studium/9_Semester/PLCTE/flask"
encoder = loaded_object['encoder']
predictor = loaded_object['predictor']
test_ids_to_test_names = loaded_object['test_ids_to_test_names']


@app.route('/data')
def test_data():
    data = {}
    with open('test_visualization_data.json') as json_file:
        data = json.load(json_file)
    print("Serve data request")
    return jsonify(data)


@socketio.on('message')
def handle_message(message):
    print('received message: ' + message)


@socketio.on('join')
def handle_my_custom_event(str):
    print('received join: ' + str)
    join_room(str)
    print('Jointed room ' + str)


@socketio.on('testreport')
def handle_my_custom_event(json):
    print('received test report: ' + str(json))
    socketio.emit('testreport', json, room='web')


@socketio.on('save')
def handle_save_event(_):
    print('received save call, starting prediction')
    t = predict_failing_tests(path, predictor, encoder, test_ids_to_test_names)
    predicted_failure_names = []
    for index in t:
        predicted_failure_names.append(test_ids_to_test_names.at[index])

    socketio.emit('predicted_failures', predicted_failure_names, room='web')

    for i in t:
        print(test_ids_to_test_names.at[i])
    
    print('starting pytest')
    cmd = 'cd /home/dominik/Studium/9_Semester/PLCTE/flask/ && . ../pytest-immediate/venv/bin/activate && pytest > /dev/null'
    subprocess.Popen(cmd, shell=True)

if __name__ == '__main__':
    socketio.run(app, port=9001, debug=True)
