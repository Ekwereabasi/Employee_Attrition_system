from django import forms
from .models import EmployeeAttritionModel

class EmployeeAttritionForm(forms.ModelForm):
    class Meta:
        model = EmployeeAttritionModel
        fields = [
            'age', 'business_travel', 'department', 'daily_rate',
            'distance_from_home', 'education', 'job_role', 'job_satisfaction',
            'monthly_income', 'work_life_balance', 'years_at_company'
        ]
