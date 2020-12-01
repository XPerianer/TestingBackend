from flask import Flask
from flask import jsonify
import subprocess
import multitasking

app = Flask(__name__)

test_data = {}

@app.route('/')
def test_data():
    return jsonify(d)


@multitasking.task
def update_test_data():
    try:
        while True:
            print("Starting Tests")
            subprocess.call('cd /home/dominik/Studium/9_Semester/PLCTE/flask/ && . venv/bin/activate ' \
                    '&& pytest -rN --json=report.json', shell=True)
    except KeyboardInterrupt:
        print('interrupted!')

if __name__ == '__main__':
    update_test_data()
    app.run(debug=True)
