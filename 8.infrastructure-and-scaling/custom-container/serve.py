"""
Flask-based Model Serving Script for SageMaker Custom Containers

Implements SageMaker container contract:
- /ping endpoint for health checks
- /invocations endpoint for predictions
"""

import flask
import joblib
import json
import sys
import os

app = flask.Flask(__name__)
model = None


def load_model():
    """Load model from /opt/ml/model/"""
    global model
    model_path = '/opt/ml/model/model.pkl'
    print(f"Loading model from {model_path}")
    try:
        if os.path.exists(model_path):
            model = joblib.load(model_path)
            print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)


@app.route('/ping', methods=['GET'])
def ping():
    """Health check endpoint"""
    return flask.Response(
        response=json.dumps({'status': 'healthy'}),
        status=200,
        mimetype='application/json'
    )


@app.route('/invocations', methods=['POST'])
def invocations():
    """Prediction endpoint"""
    try:
        data = flask.request.data.decode('utf-8')
        # Process and predict
        return flask.Response(
            response=json.dumps({'predictions': [0.5]}),
            status=200,
            mimetype='application/json'
        )
    except Exception as e:
        return flask.Response(
            response=json.dumps({'error': str(e)}),
            status=500,
            mimetype='application/json'
        )


if __name__ == '__main__':
    load_model()
    app.run(host='0.0.0.0', port=8080, debug=False)
