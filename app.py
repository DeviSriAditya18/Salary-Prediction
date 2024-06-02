import numpy as np
from flask import Flask, request, jsonify, render_template
from joblib import load 
import pickle

app = Flask(__name__)

with open('lasso_regression_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    int_features = [int(request.form.get(key, 0)) for key in ['age', 'leaves_used', 'leaves_remaining', 'ratings', 'past_exp', 'male', 'female', 'analyst', 'associate', 'director', 'manager', 'senior_analyst', 'senior_manager', 'finance', 'it', 'management', 'marketing', 'operations', 'web']]
    final_features = np.array(int_features).reshape(1, -1)  # Reshape features for prediction
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)
    return render_template('index.html', prediction_text='Employee Salary Should Be {}'.format(output))

if __name__ == "__main__":
    app.run(debug=True)

