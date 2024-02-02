# import main Flask class and request object
from flask import Flask, request
from people_counter import run
from video_capture import open_video
import json

# create the Flask app
app = Flask(__name__)

@app.route('/start-count')
def start_count():
    count=run()
    print(count)
    data={'people': count}
    response= app.response_class(response=json.dumps(data),mimetype='application/json')
    return response

@app.route('/')
def open_window():
    open_video()
    data={'people': 123}
    response=app.response_class(response=json.dumps(data),mimetype='application/json')
    return response

if __name__ == '__main__':
    # run app in debug mode on port 5002
    app.run(host='0.0.0.0', port=5002)