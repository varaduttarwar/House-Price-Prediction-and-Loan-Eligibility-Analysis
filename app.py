from flask import Flask, render_template, request
import pandas as pd
import joblib

# Initialize Flask app
app = Flask(__name__)

# Load House Price Prediction Model
house_model = joblib.load('house_price_predictor.pkl')
preprocessor = joblib.load('preprocessor.pkl')

# Load Loan Prediction Model
loan_model = joblib.load('Project.pkl')

# Feature Names for House Prediction
house_features = [
    'No of Bedrooms', 'No of Bathrooms', 'Flat Area (in Sqft)', 'Lot Area (in Sqft)',
    'No of Floors', 'Waterfront View', 'No of Times Visited', 'Condition of the House',
    'Overall Grade', 'Area of the House from Basement (in Sqft)', 'Basement Area (in Sqft)',
    'Age of House (in Years)', 'Renovated Year', 'Zipcode', 'Latitude', 'Longitude',
    'Living Area after Renovation (in Sqft)', 'Lot Area after Renovation (in Sqft)'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/house')
def house():
    return render_template('house.html', feature_names=house_features)

@app.route('/predict_house', methods=['POST'])
def predict_house():
    try:
        user_input = [float(request.form[feature]) for feature in house_features]
        user_input_df = pd.DataFrame([user_input], columns=house_features)

        transformed_input = preprocessor.transform(user_input_df)
        predicted_price = house_model.predict(transformed_input)

        return render_template('result.html', result=f'Predicted House Price: ${round(predicted_price[0], 2)}')

    except Exception as e:
        return render_template('result.html', result=f'Error: {str(e)}')

@app.route('/loan')
def loan():
    return render_template('loan.html')

@app.route('/predict_loan', methods=['POST'])
def predict_loan():
    try:
        # Get input values from the form
        gender = request.form['Gender']
        married = request.form['Married']
        dependents = int(request.form['Dependents'])
        education = request.form['Education']
        self_employed = request.form['Self_Employed']
        applicant_income = float(request.form['ApplicantIncome'])
        coapplicant_income = float(request.form['CoapplicantIncome'])
        loan_amount = float(request.form['LoanAmount'])
        loan_amount_term = float(request.form['Loan_Amount_Term'])
        credit_history = float(request.form['Credit_History'])
        property_area = request.form['Property_Area']

        # Compute derived features
        total_income = applicant_income + coapplicant_income
        loan_amount_to_income_ratio = loan_amount / total_income if total_income > 0 else 0
        credit_history_education = credit_history * (1 if education == 'Graduate' else 0)

        # Prepare DataFrame
        new_data = pd.DataFrame([{
            'Gender': 1 if gender == 'Male' else 0,
            'Married': 1 if married == 'Yes' else 0,
            'Dependents': dependents,
            'Education': 1 if education == 'Graduate' else 0,
            'Self_Employed': 1 if self_employed == 'Yes' else 0,
            'LoanAmount': loan_amount,
            'Loan_Amount_Term': loan_amount_term,
            'Credit_History': credit_history,
            'Property_Area': 2 if property_area == 'Urban' else (1 if property_area == 'Semiurban' else 0),
            'TotalIncome': total_income,
            'LoanAmountTotalIncomeRatio': loan_amount_to_income_ratio,
            'CreditHistoryEducation': credit_history_education
        }])

        # Ensure feature order matches the model
        model_columns = [
            'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
            'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 
            'CreditHistoryEducation', 'LoanAmountTotalIncomeRatio', 'TotalIncome'
        ]
        new_data = new_data[model_columns]

        # Predict loan eligibility
        prediction = loan_model.predict(new_data)

        # Convert prediction to human-readable format
        prediction_result = 'Approved' if prediction[0] == 1 else 'Not Approved'

        return render_template('result.html', result=f'Loan Status: {prediction_result}')

    except Exception as e:
        return render_template('result.html', result=f'Error: {str(e)}')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
