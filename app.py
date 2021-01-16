from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import join_room

import subprocess
import multitasking
import json
import joblib
import shlex
import pandas as pd

from pandas import DataFrame

from src import loading

from src.predict_failing_tests import predict_failing_tests

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*', ping_timeout=999999)

test_data = {}


loaded_object = joblib.load('flask_simple_decision_tree.joblib')
path = "/home/dominik/Studium/9_Semester/PLCTE/flask"
encoder = loaded_object['encoder']
predictor = loaded_object['predictor']
test_ids_to_test_names = loaded_object['test_ids_to_test_names']

# Generate Mutant Coverage Relevancy table
# We generate a map from modified file path and modified method to the failing tests, in order to quickly find this information if the user is changing the context
data = loading.load_dataset('data/flask_full.pkl')
failing_tests_lists = data.loc[data["outcome"] == False].groupby(["modified_file_path", "modified_method"])["full_name"].agg(list)
failing_tests_key_value = failing_tests_lists.map(pd.value_counts)

print(failing_tests_key_value)
relevant_tests = pd.Series()

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

@socketio.on('onDidChangeVisibleTextEditors')
def handle_did_change_visible_text_editors(textEditors):
    return
    print('received changed visible text editors: ' + str(textEditors))
    relevant_tests = pd.Series()
    for textEditor in textEditors:
        print('visible file: ' + str(textEditor['filename']))
        add = lambda a, b: a + b
        relevant_tests = relevant_tests.combine(failing_tests_key_value.at[textEditor['filename']], add, fill_value=0)

    print("Sending out the new relevancies")
    print(relevant_tests)
    socketio.emit('relevanceUpdate', relevant_tests.to_json(), room='web')

@socketio.on('onDidChangeTextEditorVisibleRanges')
def handle_did_change_text_editors_visible_ranges(visibleMethodNames):
    print('received onDidChangeTextEditorVisibleRanges call')
    print(f"data: {visibleMethodNames}")
    method_to_failing_tests_count = failing_tests_key_value.at[visibleMethodNames["filename"]]
    relevant_tests = pd.Series()
    add = lambda a, b: a + b
    for method in visibleMethodNames["visibleMethodNames"]:
        if method in method_to_failing_tests_count.index:
            relevant_tests = relevant_tests.combine(method_to_failing_tests_count.at[method], add, fill_value=0)

    print("Sending out the new relevancies")
    print(relevant_tests)
    socketio.emit('relevanceUpdate', relevant_tests.to_json(), room='web')
        
@socketio.on('save')
def handle_save_event(_):
    print('received save call, starting prediction')
    t = predict_failing_tests(path, predictor, encoder, test_ids_to_test_names)
    predicted_failure_names = []
    for index in t:
        predicted_failure_names.append(test_ids_to_test_names.at[index])

    test_order = predicted_failure_names
    print(f"Test Order: {test_order}")
    print('starting pytest')
    cmd = f"cd /home/dominik/Studium/9_Semester/PLCTE/flask/ && . ../pytest-immediate/venv/bin/activate && pytest --test_ordering {shlex.quote(json.dumps(test_order))} > /dev/null"
    print(cmd)
    p = subprocess.Popen(cmd, shell=True)
    socketio.emit('predicted_failures', predicted_failure_names, room='web')

if __name__ == '__main__':
    socketio.run(app, port=9001, debug=True)
