# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib

# Load the dataset
data = pd.read_csv('../Training/WA_Fn-UseC_-HR-Employee-Attrition.csv')

# Define the features and target
X = data.drop(columns=['Attrition'])
y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert Attrition to binary (1 for Yes, 0 for No)

# Define categorical and numeric features
categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole']

# Preprocessing pipelines for both numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')  # Drop one to avoid collinearity

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Build the model pipeline with cost-sensitive learning
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train the model
pipeline.fit(X_train, y_train)

# Predict on test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model for deployment in Django
joblib.dump(pipeline, 'attrition_model_cost_S.pkl')
