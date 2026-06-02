import streamlit as st
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Page configuration
st.set_page_config(
    page_title="Fraud Detection System",
    page_icon="🔍",
    layout="centered"
)

# Custom CSS styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0d1117;
        color: white;
    }
    .stButton>button {
        width: 100%;
        padding: 15px;
        font-size: 16px;
        font-weight: bold;
        border-radius: 10px;
        background-color: #1f77b4;
        color: white;
    }
    .title {
        text-align: center;
        color: #1f77b4;
        font-size: 40px;
        font-weight: bold;
    }
    .subtitle {
        text-align: center;
        color: #aaaaaa;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and scaler
model = joblib.load('fraud_model.pkl')
scaler = joblib.load('scaler.pkl')

# Title
st.markdown('<p class="title">🔍 Fraud Detection System</p>',
            unsafe_allow_html=True)
st.markdown('<p class="subtitle">Enter transaction details below to detect fraud</p>',
            unsafe_allow_html=True)
st.markdown("---")

# Input form
st.markdown("### 📋 Enter Transaction Details")

col1, col2 = st.columns(2)

with col1:
    step = st.number_input(
        "Step (Time Unit)",
        min_value=1,
        max_value=744,
        value=1,
        step=1,
        help="Represents the hour of the transaction (1-744)"
    )

    transaction_type = st.selectbox(
        "Transaction Type",
        options=['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'],
        help="Select the type of transaction"
    )

    amount = st.number_input(
        "Amount (₦)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Enter the transaction amount"
    )

    old_balance_orig = st.number_input(
        "Old Balance Origin (₦)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Sender account balance before transaction"
    )

with col2:
    new_balance_orig = st.number_input(
        "New Balance Origin (₦)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Sender account balance after transaction"
    )

    old_balance_dest = st.number_input(
        "Old Balance Destination (₦)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Receiver account balance before transaction"
    )

    new_balance_dest = st.number_input(
        "New Balance Destination (₦)",
        min_value=0.0,
        value=0.0,
        step=0.01,
        help="Receiver account balance after transaction"
    )

st.markdown("---")

# Predict button
if st.button("🔍 Detect Fraud", use_container_width=True):

    # Encode transaction type
    le = LabelEncoder()
    le.fit(['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER'])
    type_encoded = le.transform([transaction_type])[0]

    # Create input array with correct feature order
    input_data = np.array([[step, type_encoded, amount,
                            old_balance_orig, new_balance_orig,
                            old_balance_dest, new_balance_dest]])

    # Create dataframe with correct column names for scaler
    input_df = pd.DataFrame(input_data, columns=[
        'step', 'type', 'amount',
        'oldbalanceOrg', 'newbalanceOrig',
        'oldbalanceDest', 'newbalanceDest'
    ])

    # Scale input correctly using saved scaler
    input_scaled = scaler.transform(input_df)

    # Predict
    prediction = model.predict(input_scaled)

    # Display transaction summary
    st.markdown("### 📋 Transaction Summary")
    summary = pd.DataFrame({
        'Feature': ['Step', 'Type', 'Amount',
                   'Old Balance Origin', 'New Balance Origin',
                   'Old Balance Destination', 'New Balance Destination'],
        'Value': [step, transaction_type,
                 f"₦{amount:,.2f}",
                 f"₦{old_balance_orig:,.2f}",
                 f"₦{new_balance_orig:,.2f}",
                 f"₦{old_balance_dest:,.2f}",
                 f"₦{new_balance_dest:,.2f}"]
    })
    st.dataframe(summary, use_container_width=True)

    # Display result
    st.markdown("### 🤖 XGBoost Model Prediction")
    if prediction[0] == 0:
        st.success('✅ LEGITIMATE Transaction')
    else:
        st.error('⚠️ FRAUD Detected')

st.markdown("---")
st.markdown('<p class="subtitle">Powered by XGBoost Machine Learning Model</p>',
            unsafe_allow_html=True)