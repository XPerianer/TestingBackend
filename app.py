from flask import Flask
from flask import jsonify
from flask_cors import CORS
from flask_socketio import SocketIO
from flask_socketio import join_room

import subprocess
import multitasking
import json

app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins='*')

test_data = {}

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

if __name__ == '__main__':
    socketio.run(app, port=9001, debug=True)
