import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE
import joblib

# Load the dataset
data = pd.read_csv('Training/filtered_employee_attrition_with_attrition.csv')

# Define the features and target
X = data.drop(columns=['Attrition'])
y = data['Attrition'].apply(lambda x: 1 if x == 'Yes' else 0)  # Convert Attrition to binary (1 for Yes, 0 for No)

# Define categorical and numeric features
categorical_features = ['BusinessTravel', 'Department', 'EducationField', 'Gender', 'JobRole', 'MaritalStatus', 'OverTime']
numeric_features = ['Age', 'DailyRate', 'DistanceFromHome', 'Education', 'JobLevel', 'JobSatisfaction', 'MonthlyIncome',
                    'WorkLifeBalance', 'YearsAtCompany', 'YearsInCurrentRole']

# Preprocessing pipelines for both numeric and categorical features
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(drop='first')  # Drop one category to avoid collinearity

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# First, preprocess the training data using the preprocessor before applying SMOTE
X_train_transformed = preprocessor.fit_transform(X_train)

# Apply SMOTE to oversample the minority class in the training set
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_transformed, y_train)

# Build the final model pipeline (without preprocessing, as it’s already applied above)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Preprocess the test data
X_test_transformed = preprocessor.transform(X_test)

# Predict on test set
y_pred = model.predict(X_test_transformed)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Save the model for deployment in Django
joblib.dump((preprocessor, model), 'attrition_model_smote.pkl')
