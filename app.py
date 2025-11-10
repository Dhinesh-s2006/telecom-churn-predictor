from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# --- 1. Load the Trained Model (NO SCALER NEEDED) ---
try:
    # Load the model only. The Decision Tree was trained on unscaled data.
    MODEL = joblib.load('churn_model.pkl')
    # NOTE: The preprocessor.pkl file is NOT loaded for this model, as scaling is skipped.
    print("Decision Tree Model loaded successfully (Scaling skipped).") 
except FileNotFoundError:
    print("Error: churn_model.pkl not found. Please ensure the final Decision Tree model is saved.")
    exit()

# --- 2. Initialize Flask App ---
app = Flask(__name__)
CORS(app) 

# --- 2. Define Feature Constants (MUST MATCH TRAINING) ---
# NOTE: SCALING_COLS REMOVED
BINARY_COLS = ["SeniorCitizen", "Partner", "Dependents", "PaperlessBilling"] 
ONE_HOT_COLS = ["Contract", "PaymentMethod"]

# The final column order used for training (crucial!)
FINAL_COLUMN_ORDER = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PaperlessBilling', 
    'MonthlyCharges', 'TotalCharges', 'NumServices', 'SpendingLevel', 
    'LoyaltyFlag', 'TenureGroup', 'Expected_Total_Charges', 'Charge_Deviation', 
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)', 
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# --- Global Constants for Feature Engineering (MUST MATCH TRAINING DATA!) ---
TRAINING_MONTHLY_CHARGES_MEDIAN = 64.85189431704886 
TOTAL_CHARGES_QUANTILES = [18.80, 750.00, 2900.00, 8684.80] # Using 4 points for the 3 bins
TENURE_BINS = [0, 12, 36, 60, 100]
TENURE_LABELS = ['New', 'Mid-Term', 'Loyal', 'Veteran']
SERVICE_COLS = ["MultipleLines", "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]


# --- 3. Feature Construction Function (COMPLETE MANUAL METHOD) ---
def construct_features(data):
    """Applies all the custom feature engineering and manual encoding steps."""
    
    df_single = pd.DataFrame([data]) 

    # Calculate base numeric values needed for derived features
    tenure = df_single['tenure'].iloc[0]
    monthly_charges = df_single['MonthlyCharges'].iloc[0]
    total_charges = df_single['TotalCharges'].iloc[0]

    # A. NumServices 
    df_single["NumServices"] = df_single[SERVICE_COLS].apply(lambda x: sum(x == "Yes"), axis=1)

    # B. Spending Level 
    labels_map = {1: 'Low', 2: 'Medium', 3: 'High'}
    df_single["SpendingLevel"] = pd.cut(
        df_single["TotalCharges"], 
        bins=TOTAL_CHARGES_QUANTILES, 
        labels=labels_map.values(), 
        right=True, 
        include_lowest=True
    ).astype(str)

    # C. LoyaltyFlag 
    df_single["LoyaltyFlag"] = np.where(
        (df_single["tenure"] > 24) & (df_single["MonthlyCharges"] < TRAINING_MONTHLY_CHARGES_MEDIAN),
        1,
        0
    )

    # D. TenureGroup
    df_single['TenureGroup'] = pd.cut(
        df_single['tenure'],
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        right=True,
        include_lowest=True
    ).astype(str)

    # E. Charge Deviation
    df_single['Expected_Total_Charges'] = monthly_charges * tenure
    df_single['Charge_Deviation'] = total_charges - df_single['Expected_Total_Charges']
    df_single['Charge_Deviation'] = np.where(df_single['tenure'] == 0, 0, df_single['Charge_Deviation'])

    # --- Manual Encoding Replication ---
    
    # 1. Binary Mapping (Yes/No to 1/0)
    for col in BINARY_COLS:
        df_single[col] = df_single[col].map({"No": 0, "Yes": 1}) 
        
    # 2. Ordinal Encoding 
    spending_map = {'Low': 0, 'Medium': 1, 'High': 2}
    tenure_map = {'New': 0, 'Mid-Term': 1, 'Loyal': 2, 'Veteran': 3}
    df_single["SpendingLevel"] = df_single["SpendingLevel"].map(spending_map)
    df_single["TenureGroup"] = df_single["TenureGroup"].map(tenure_map)
    
    # 3. One-Hot Encoding 
    df_single = pd.get_dummies(df_single, columns=ONE_HOT_COLS, drop_first=True)
    
    # 4. Final Column Alignment
    for col in FINAL_COLUMN_ORDER:
        if col not in df_single.columns:
            df_single[col] = 0
            
    # 5. Select and reorder the final feature set
    df_single = df_single[FINAL_COLUMN_ORDER] 
    
    return df_single


# --- 4. Define the Prediction Endpoint ---
@app.route('/predict_churn', methods=['POST'])
def predict():
    raw_data = request.get_json(force=True)
    
    # Validation (Check for all RAW columns needed by the script)
    required_raw_keys = ['tenure', 'MonthlyCharges', 'TotalCharges', 'Contract', 'Partner', 'Dependents', 
                         'PaperlessBilling', 'PaymentMethod', 'SeniorCitizen', *SERVICE_COLS]
    
    if not all(key in raw_data for key in required_raw_keys):
         missing = [key for key in required_raw_keys if key not in raw_data]
         return jsonify({'error': f'Missing required input fields: {", ".join(missing)}'}), 400

    try:
        # 1. Construct and encode all features (THIS IS THE ONLY PROCESSING STEP)
        # The Decision Tree was trained on this unscaled format.
        X_processed = construct_features(raw_data)
        
        # 2. Predict probability
        # The model expects a NumPy array, so use .values
        proba = MODEL.predict_proba(X_processed.values)[0]
        churn_prob = proba[1] 

        # 3. Return the result
        return jsonify({
            'churn_probability': float(churn_prob),
            'status': 'success'
        })

    except Exception as e:
        # Log the detailed error for debugging, return general error to client
        print(f"Prediction Error: {e}")
        return jsonify({'error': 'An internal server error occurred during prediction.'}), 500

# --- 5. Run the Server ---
if __name__ == '__main__':
    app.run(debug=True, port=5000)