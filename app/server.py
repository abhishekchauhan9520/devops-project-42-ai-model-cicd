import os
import joblib
import pandas as pd
from flask import Flask, jsonify, request

MODEL_PATH = os.getenv('MODEL_PATH', '/models/model.joblib')
MODEL_VERSION = os.getenv('MODEL_VERSION', 'local')
model = joblib.load(MODEL_PATH)
app = Flask(__name__)
FEATURES = [f'feature_{i}' for i in range(6)]

@app.get('/health')
def health():
    return jsonify({'status': 'ok', 'model_version': MODEL_VERSION})

@app.post('/predict')
def predict():
    payload = request.get_json(force=True)
    rows = payload.get('instances', [])
    if not rows:
        return jsonify({'error': 'instances is required'}), 400
    df = pd.DataFrame(rows)
    if list(df.columns) != FEATURES:
        return jsonify({'error': f'exact feature order required: {FEATURES}'}), 400
    pred = model.predict(df[FEATURES]).tolist()
    return jsonify({'predictions': pred, 'model_version': MODEL_VERSION})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
