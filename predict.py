import pickle


import xgboost as xgb
from flask import Flask, jsonify, request


model_file = 'diabetes_model.bin'

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)

app = Flask('diabetes_predictor')


@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(silent=True)

    if data is None:
        return jsonify({'error': 'Request body must be valid JSON.'}), 400

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list) or not all(isinstance(item, dict) for item in data):
        return jsonify({'error': 'Expected a JSON object or list of objects with feature names.'}), 400

    X = dv.transform(data)
    dmatrix = xgb.DMatrix(X)
    y_pred = model.predict(dmatrix)
    diabetes_risk = [int(value >= 0.5) for value in y_pred.tolist()]
    result = {
        'predictions': diabetes_risk[0] if len(diabetes_risk) == 1 else diabetes_risk
    }
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)