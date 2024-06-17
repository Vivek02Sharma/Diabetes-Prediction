from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load scaler and model outside of the route function to load them once
with open('pickle file/scaler.pkl', 'rb') as file1:
    scaler = pickle.load(file1)

with open('pickle file/model.pkl', 'rb') as file2:
    model = pickle.load(file2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/submit', methods=['POST','GET'])
def submit():
    if request.method == 'POST':
        # Get form data and validate numeric input
        pregnancies = float(request.form['pregnancies'])
        glucose = float(request.form['glucose'])
        blood_pressure = float(request.form['bloodPressure'])
        skin_thickness = float(request.form['skinThickness'])
        insulin = float(request.form['insulin'])
        bmi = float(request.form['bmi'])
        diabetes_pedigree = float(request.form['diabetesPedigree'])
        age = float(request.form['age'])

    # Create a numpy array for the input data
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness,
                            insulin, bmi, diabetes_pedigree, age]])

    # Scale the input data
    X_scaled = scaler.transform(input_data)

    # Predict using the model
    prediction = model.predict(X_scaled)
    if prediction[0] == 1:
        result = "diabetic"
    else:
        result = "not diabetic"

    return render_template('result.html', pregnancies=pregnancies, glucose=glucose, blood_pressure=blood_pressure,skin_thickness=skin_thickness, insulin=insulin, bmi=bmi,diabetes_pedigree=diabetes_pedigree, age=age, result=result)

if __name__ == '__main__':
    app.run(debug=True, host='localhost')
