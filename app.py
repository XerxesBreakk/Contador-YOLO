# import main Flask class and request object
from flask import Flask, request
from people_counter import run
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

if __name__ == '__main__':
    # run app in debug mode on port 5000
    app.run(debug=True, port=5000)