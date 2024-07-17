from flask import Flask, request, render_template, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from dateutil import parser

app = Flask(__name__)

# Define the paths for the model, scaler, and selector
MODEL_PATH = r'D:\HotelReservationUsingFlask\model\model2.pkl'
SCALER_PATH = r'D:\HotelReservationUsingFlask\model\scaler2.pkl'
SELECTOR_PATH = r'D:\HotelReservationUsingFlask\model\selector2.pkl'

# Load the pre-trained model, scaler, and selector
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
vt_selector = joblib.load(SELECTOR_PATH)


@app.route('/')
def home():
    return render_template('predict_form.html')


@app.route('/predict', methods=['POST'])
def predict():
    # Extract data from form
    data = request.form.to_dict()
    df = pd.DataFrame([data])

    # Preprocess the data as per your training pipeline
    processed_data = preprocess_input(df)

    # Predict using the pre-trained model
    prediction = model.predict(processed_data)

    # Redirect based on prediction
    if prediction[0] == 1:
        return redirect(url_for('canceled'))
    else:
        return redirect(url_for('not_canceled'))


def preprocess_input(df):
    # Convert relevant columns to numeric values
    numeric_features = ['number of adults', 'number of children', 'number of weekend nights',
                        'number of week nights', 'lead time', 'P-C', 'P-not-C', 'average price', 'special requests']

    for feature in numeric_features:
        df[feature] = pd.to_numeric(df[feature])

    # Convert 'car parking space' and 'repeated' to 0 or 1
    df['car parking space'] = df['car parking space'].apply(lambda x: 1 if x == 'on' else 0)
    df['repeated'] = df['repeated'].apply(lambda x: 1 if x == 'on' else 0)

    # Function to parse dates with mixed formats
    def parse_dates(date_str):
        try:
            return parser.parse(date_str)
        except ValueError:
            return pd.NaT

    # Apply the function to parse the date column
    df['date of reservation'] = df['date of reservation'].apply(parse_dates)

    # Drop rows with unparseable dates
    df = df.dropna(subset=['date of reservation'])

    # Extract year, month, and day information
    df['reservation_year'] = df['date of reservation'].dt.year
    df['reservation_month'] = df['date of reservation'].dt.month
    df['reservation_day'] = df['date of reservation'].dt.day

    # Drop the original date column as we have extracted useful features from it
    df = df.drop(columns=['date of reservation'])

    # One-hot encode the categorical columns
    df = pd.get_dummies(df, columns=['type of meal', 'room type', 'market segment type'], drop_first=True, dtype=int)

    # Ensure all expected columns are present
    expected_columns = scaler.feature_names_in_
    for col in expected_columns:
        if col not in df.columns:
            df[col] = 0

    # Reorder columns to match the training data
    df = df[expected_columns]

    # Scale the numeric features
    df_scaled = scaler.transform(df)

    # Apply the variance threshold selector
    df_selected = vt_selector.transform(df_scaled)

    return df_selected



@app.route('/not_canceled')
def not_canceled():
    return render_template('not_canceled.html')


@app.route('/canceled')
def canceled():
    return render_template('canceled.html')
if __name__ == "__main__":
    app.run(debug=True)