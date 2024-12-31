import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
import plotly.graph_objects as go
import tempfile
import base64
from io import StringIO
import os



def show_predictions():

    def process_csv_data(href: str) -> pd.DataFrame:
        """Process CSV data from base64 encoded href string."""
        try:
            base64_data = href.split("base64,")[1].split('"')[0]
            csv_data = base64.b64decode(base64_data).decode("utf-8")
            return pd.read_csv(StringIO(csv_data))
        except (IndexError, ValueError) as e:
            st.error(f"Failed to process CSV data: {str(e)}")
            return pd.DataFrame()  # Return an empty DataFrame on error

    # Title and instructions
    st.title("Forecasting AI")
    st.write("Upload a CSV file with at least two columns: a date column and a numeric column for predictions.")

    # Sidebar for model parameters and user input
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Save the uploaded file temporarily and store the path
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            st.session_state['uploaded_file_path'] = tmp_file.name

    # Access and read data if file path is available
    if st.session_state.get('uploaded_file_path'):
        data = pd.read_csv(st.session_state['uploaded_file_path'])

    elif 'Link' in st.session_state and st.session_state['Link']:
        # Process data from the base64 encoded href if no file is uploaded
        data = process_csv_data(st.session_state['Link'])
    else:
        data = pd.DataFrame()  # Empty DataFrame if no data is available

    if not data.empty:
        # Select columns for date and target
        date_column = st.sidebar.selectbox("Select Date Column", options=data.columns.tolist(), key="date_column")
        target_column = st.sidebar.selectbox("Select Target Column", options=data.columns.tolist(), key="target_column")

        # Convert selected date column to datetime
        data[date_column] = pd.to_datetime(data[date_column], errors='coerce')
        data = data.dropna(subset=[date_column])  # Remove invalid dates

        # Create two columns for side-by-side display with equal width
        col1, col2 = st.columns([1, 1])  # Adjust the ratio as needed

        # Display data preview in the first column
        with col1:
            st.subheader("Data Preview")
            st.dataframe(data.head())

        # Display summary statistics in the second column
        with col2:
            st.subheader("Summary Statistics")
            st.write(data.describe())

        # Model and forecast settings
        model_option = st.sidebar.selectbox("Select Forecasting Model", options=["Prophet", "ARIMA"])
        periods = st.sidebar.number_input("Forecast Period (days)", min_value=1, value=10)

        # Forecast button
        if st.sidebar.button("Generate Forecast"):
            if model_option == "Prophet":
                # Prophet model setup
                model = Prophet()
                model.fit(data[[date_column, target_column]].rename(columns={date_column: 'ds', target_column: 'y'}))
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)
                st.session_state['forecast_results'] = forecast

            elif model_option == "ARIMA":
                # ARIMA model setup
                order = st.sidebar.selectbox("Select ARIMA Order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)])
                model = ARIMA(data[target_column], order=order)
                model_fit = model.fit()
                forecast = model_fit.forecast(steps=periods)
                forecast_dates = pd.date_range(start=data[date_column].max() + pd.Timedelta(days=1), periods=periods)
                st.session_state['forecast_results'] = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast})

            # Display forecast results
            if 'forecast_results' in st.session_state:
                st.subheader("Forecast Results")
                st.dataframe(st.session_state['forecast_results'])  # This will allow scrolling if the DataFrame is large
                
                # Plotting the forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data[date_column], y=data[target_column], mode='lines', name='Original Data'))
                
                if model_option == "Prophet":
                    fig.add_trace(go.Scatter(x=st.session_state['forecast_results']['ds'],
                                             y=st.session_state['forecast_results']['yhat'],
                                             mode='lines', name='Forecast'))
                else:
                    fig.add_trace(go.Scatter(x=st.session_state['forecast_results']['ds'],
                                             y=st.session_state['forecast_results']['yhat'],
                                             mode='lines', name='ARIMA Forecast'))

                # Configure layout
                fig.update_layout(
                    title="Forecasted Data",
                    xaxis_title="Date",
                    yaxis_title=target_column,
                    template="plotly_white"
                )
                
                st.plotly_chart(fig, use_container_width=True)

            # Clear Data button
        if st.sidebar.button("Clear Data"):
            # Clear session state variables related to data
            if 'uploaded_file_path' in st.session_state:
                # Remove the temporary file
                del st.session_state['uploaded_file_path']
            if 'Link' in st.session_state:
                del st.session_state['Link']
            if 'forecast_results' in st.session_state:
                del st.session_state['forecast_results']






