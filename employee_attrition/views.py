
from django.shortcuts import render
from django.http import JsonResponse
import joblib
import numpy as np
import pandas as pd  # Added to handle DataFrame creation

# Load the pre-trained model and preprocessor
model_file = 'employee_attrition/attrition_model_smote.pkl'
preprocessor, model = joblib.load(model_file)

def predict_attrition(request):
    if request.method == 'POST':
        # Collect input data from the form
        input_data = {
            'Age': request.POST.get('Age'),
            'DailyRate': request.POST.get('DailyRate'),
            'DistanceFromHome': request.POST.get('DistanceFromHome'),
            'Education': request.POST.get('Education'),
            'JobLevel': request.POST.get('JobLevel'),
            'JobSatisfaction': request.POST.get('JobSatisfaction'),
            'MonthlyIncome': request.POST.get('MonthlyIncome'),
            'WorkLifeBalance': request.POST.get('WorkLifeBalance'),
            'YearsAtCompany': request.POST.get('YearsAtCompany'),
            'YearsInCurrentRole': request.POST.get('YearsInCurrentRole'),
            'BusinessTravel': request.POST.get('BusinessTravel'),
            'Department': request.POST.get('Department'),
            'EducationField': request.POST.get('EducationField'),
            'Gender': request.POST.get('Gender'),
            'JobRole': request.POST.get('JobRole'),
            'MaritalStatus': request.POST.get('MaritalStatus'),
            'OverTime': request.POST.get('OverTime')
        }

        # Convert the input data into a pandas DataFrame for proper preprocessing
        input_df = pd.DataFrame([input_data])

        # Preprocess the input
        preprocessed_input = preprocessor.transform(input_df)

        # Make a prediction
        prediction = model.predict(preprocessed_input)
        result = 'Likely to leave' if prediction[0] == 1 else 'Likely to stay'

        # Return the prediction and the input values to the template
        return render(request, 'employee_attrition/predict_result.html', {
            'age': input_data['Age'],
            'daily_rate': input_data['DailyRate'],
            'distance_from_home': input_data['DistanceFromHome'],
            'monthly_income': input_data['MonthlyIncome'],
            'business_travel': input_data['BusinessTravel'],
            'department': input_data['Department'],
            'education_field': input_data['EducationField'],
            'gender': input_data['Gender'],
            'job_role': input_data['JobRole'],
            'marital_status': input_data['MaritalStatus'],
            'job_satisfaction': input_data['JobSatisfaction'],
            'job_level': input_data['JobLevel'],
            'work_life_balance': input_data['WorkLifeBalance'],
            'years_at_company': input_data['YearsAtCompany'],
            'years_in_current_role': input_data['YearsInCurrentRole'],
            'overtime': input_data['OverTime'],
            'prediction': result
        })

    return render(request, 'employee_attrition/predict_attrition.html')

