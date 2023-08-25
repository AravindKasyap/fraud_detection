import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset
data = pd.read_csv('new_fraud_dataset.csv')  # Replace with the actual path to your dataset

# Preprocessing code
# Specify the categorical column you want to fill null values for
categorical_column = 'type'

# Calculate the mode of the categorical column
mode_value = data[categorical_column].mode()[0]

# Fill null values in the categorical column with the mode
data[categorical_column].fillna(mode_value, inplace=True)

data = data.drop(columns=['isFlaggedFraud'])

# Columns to standardize and replace
columns_to_standardize = ['oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']

# Create a new DataFrame with only the columns to be standardized
data_to_standardize = data[columns_to_standardize]

# Initialize the StandardScaler
scaler = StandardScaler()

# Fit and transform the selected columns
standardized_data = scaler.fit_transform(data_to_standardize)

# Create a DataFrame from the standardized data
standardized_df = pd.DataFrame(standardized_data, columns=columns_to_standardize)

# Replace the selected columns in the original DataFrame
data[columns_to_standardize] = standardized_df

fraud = data[data['isFraud'] == 1]
normal = data[data['isFraud'] == 0]

data = pd.get_dummies(data, columns=['type'], prefix='type', drop_first=True)
columns_to_drop = ['nameOrig', 'nameDest']
data.drop(columns=columns_to_drop, inplace=True)

# Handle missing values by imputing with mean (you can adjust this strategy)
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(data)

# Convert imputed data back to a DataFrame
data_imputed = pd.DataFrame(data_imputed, columns=data.columns)

# Specify the target variable and feature columns
target_column = 'isFraud'
feature_columns = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest',
                   'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER']

# Split the data into features (X) and target variable (y)
X = data_imputed[feature_columns]
y = data_imputed[target_column]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Isolation Forest model
def train_isolation_forest(X_train):
    isolation_forest = IsolationForest(contamination=0.05, random_state=42)
    isolation_forest.fit(X_train)
    return isolation_forest

# Train the Isolation Forest model
isolation_forest = train_isolation_forest(X_train)

# Save the trained model
model_filename = 'isolation_forest_model.joblib'
joblib.dump(isolation_forest, model_filename)

# Load the trained Isolation Forest model
model_filename = 'isolation_forest_model.joblib'
isolation_forest = joblib.load(model_filename)

# Load the scaler info for scaling user input
scaler_info = {
    'amount': {'mean': 135960.73, 'std': 311907.25},
    'oldbalanceOrg': {'mean': 839585.09, 'std': 2930765.27},
    'newbalanceOrig': {'mean': 862786.81, 'std': 2992454.30},
    'oldbalanceDest': {'mean': 1100700.49, 'std': 3399183.41},
    'newbalanceDest': {'mean': 1224996.40, 'std': 3482463.80}
}

# Function to preprocess input data
def preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type):
    scaled_data = {
        'amount': (amount - scaler_info['amount']['mean']) / scaler_info['amount']['std'],
        'oldbalanceOrg': (oldbalanceOrg - scaler_info['oldbalanceOrg']['mean']) / scaler_info['oldbalanceOrg']['std'],
        'newbalanceOrig': (newbalanceOrig - scaler_info['newbalanceOrig']['mean']) / scaler_info['newbalanceOrig']['std'],
        'oldbalanceDest': (oldbalanceDest - scaler_info['oldbalanceDest']['mean']) / scaler_info['oldbalanceDest']['std'],
        'newbalanceDest': (newbalanceDest - scaler_info['newbalanceDest']['mean']) / scaler_info['newbalanceDest']['std'],
        'type_CASH_OUT': type == 'CASH_OUT',
        'type_DEBIT': type == 'DEBIT',
        'type_PAYMENT': type == 'PAYMENT',
        'type_TRANSFER': type == 'TRANSFER'
    }
    data = pd.DataFrame(scaled_data, index=[0])
    return data

# Streamlit app
def main():
    st.title("Credit Card Fraud Detection")

    amount = st.number_input("Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance Orig", value=5000.0)
    newbalanceOrig = st.number_input("New Balance Orig", value=4000.0)
    oldbalanceDest = st.number_input("Old Balance Dest", value=0.0)
    newbalanceDest = st.number_input("New Balance Dest", value=1000.0)
    type = st.selectbox("Type", ['CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])

    if st.button("Predict"):
        input_data = preprocess_input(amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type)
        prediction = isolation_forest.predict(input_data)
        result = "Fraudulent" if prediction[0] == 1 else "Legitimate"
        st.success(f"The transaction is predicted to be: {result}")

if __name__ == "__main__":
    main()