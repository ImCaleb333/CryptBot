import streamlit as st
import numpy as np
import joblib  # Assuming you saved your model using joblib


# Disclaimer at the top of the app
st.markdown("""
    <h3 style="color:red;">Disclaimer:</h3>
    <p>This is a test project for educational purposes only. The cryptocurrency closing price predictions provided by this app are 
    based on a machine learning model and are not guaranteed to be accurate. This tool is not intended for real-world trading or 
    investment decisions. Please do your own research and consult with a financial advisor before making any trading decisions.</p>
    <hr>
    """, unsafe_allow_html=True)

# Load your trained model
model = joblib.load('Model')  # Adjust the path to where your model is saved

# Streamlit App Layout
st.title("Cryptocurrency Closing Price Prediction")

st.header("Enter Open, High, and Low prices:")

# User inputs for Open, High, and Low prices (these are numbers you type in)
open_price = st.text_input('Open Price')  # Example default value
high_price = st.text_input('High Price')  # Example default value
low_price = st.text_input('Low Price')  # Example default value

# Validation to ensure inputs are numeric
if open_price and high_price and low_price:  # Check if inputs are not empty
    try:
        open_price = float(open_price)
        high_price = float(high_price)
        low_price = float(low_price)
        
        # Ensure prices are greater than zero (or any other logical check you need)
        if open_price > 0 and high_price > 0 and low_price >= 0:
            # Prepare the input data for prediction
            input_data = np.array([[open_price, high_price, low_price]])
            
            # Make the prediction
            predicted_close = model.predict(input_data)
            
            # Display the result
            st.subheader(f"Predicted Closing Price: ${predicted_close[0]}")
        else:
            st.error("Please enter valid values for Open, High, and Low prices.")
    except ValueError:
        st.error("Please enter valid numeric values for Open, High, and Low prices.")
else:
    st.error("Please fill in all the fields with valid numbers.")