import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import plotly.graph_objects as go
import tempfile

def show_predictions():
    # # Page configuration
    # st.set_page_config(layout="wide", page_title="DataManagement AI", page_icon="img/bkofkgl.png",
    #                    menu_items={'Get Help': 'mailto:john@example.com',
    #                                'About': "#### This is DataManagement cool app!"})

    # Title and Instructions
    st.title("Predictions Page")
    st.write("Upload a CSV file containing two columns: a date column ('ds') and a numeric column ('y') for predictions.")
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily and store the path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            st.session_state['uploaded_file_path'] = tmp_file.name

    # Access and read data if file path is available
    if st.session_state.get('uploaded_file_path'):
        data = pd.read_csv(st.session_state['uploaded_file_path'])
        st.subheader("Uploaded Data")
        st.write(data.head())
        
        # Check columns and preprocess
        if 'ds' in data.columns and 'y' in data.columns:
            data['ds'] = pd.to_datetime(data['ds'], errors='coerce')  # Convert 'ds' to datetime
            data = data.dropna(subset=['ds'])  # Remove invalid dates
            
            st.subheader("Data with DateTime")
            st.write(data.head())
            st.subheader("Summary Statistics")
            st.write(data.describe())
            
            # Model selection and forecast period input
            st.session_state['model_option'] = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "Exponential Smoothing"], index=0)
            st.session_state['periods'] = st.slider("Select number of future periods to forecast", 1, 365, 30)
            
            # Forecast button
            if st.button("Forecast"):
                if st.session_state['model_option'] == "Prophet":
                    model = Prophet()
                    model.fit(data)
                    future = model.make_future_dataframe(periods=st.session_state['periods'])
                    forecast = model.predict(future)
                    st.session_state['forecast_results'] = forecast

                elif st.session_state['model_option'] == "ARIMA":
                    order = st.selectbox("Select ARIMA Order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)])
                    model = ARIMA(data['y'], order=order)
                    model_fit = model.fit()
                    forecast = model_fit.forecast(steps=st.session_state['periods'])
                    forecast_dates = pd.date_range(start=data['ds'].max() + pd.Timedelta(days=1), periods=st.session_state['periods'])
                    st.session_state['forecast_results'] = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast})

               
