import pickle
import pandas as pd
import streamlit as st

# Load the trained model and scalers
model = pickle.load(open('your_trained_model.pkl', 'rb'))
scaler_amount = pickle.load(open('your_amount_scaler.pkl', 'rb'))
scaler_time = pickle.load(open('your_time_scaler.pkl', 'rb'))

# Streamlit app setup
st.title('Credit Card Fraud Detection')

# File uploader

def classify_transactions(new_data, model):
    new_data['Amount_scaled'] = scaler_amount.transform(new_data[['Amount']])
    new_data['Time_scaled'] = scaler_time.transform(new_data[['Time']])
    new_data = new_data.drop(['Amount', 'Time'], axis=1)
    new_data['anomaly'] = model.predict(new_data)
    new_data['anomaly'] = new_data['anomaly'].apply(lambda x: 1 if x == -1 else 0)
    return new_data

# Streamlit app
st.title('Credit Card Fraud Detection')

st.write('Upload a CSV file with credit card transactions:')

uploaded_file = st.file_uploader("Choose a file", type="csv")

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    
    # Ensure the file has 'Amount' and 'Time' columns
    if 'Amount' not in data.columns or 'Time' not in data.columns:
        st.error("The uploaded file must contain 'Amount' and 'Time' columns.")
    else:
        # Classify the transactions
        results = classify_transactions(data, model)
        
        st.write('Results:')
        st.write(results)
        
        # Display some visualizations if needed
        st.subheader('Fraudulent Transactions')
        fraudulent_transactions = results[results['anomaly'] == 1]
        st.write(fraudulent_transactions)