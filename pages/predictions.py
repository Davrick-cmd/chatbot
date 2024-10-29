import streamlit as st
import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import plotly.graph_objects as go

# Page configuration
st.set_page_config(layout="wide", page_title="DataManagement AI", page_icon="img/bkofkgl.png",
                   menu_items={'Get Help': 'mailto:john@example.com',
                               'About': "#### This is DataManagement cool app!"})

# Check if the user is logged in
st.write(st.session_state)
if "authenticated" not in st.session_state or not st.session_state.username:
    st.error("You need to log in to access this page.")
    st.switch_page('app.py')
    st.stop()  # Stop execution of this page

st.title("Predictions Page")

st.write(
    "Upload a CSV file containing two columns: a date column ('ds') and a numeric column ('y') for predictions."
)

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the CSV file
    data = pd.read_csv(uploaded_file)

    # Show the raw data
    st.subheader("Raw Data")
    st.write(data.head())

    # Check if the necessary columns 'ds' and 'y' are present
    if 'ds' in data.columns and 'y' in data.columns:
        # Convert 'ds' column to datetime
        data['ds'] = pd.to_datetime(data['ds'], errors='coerce')

        # Remove rows with missing or invalid dates
        data = data.dropna(subset=['ds'])

        st.subheader("Data with DateTime")
        st.write(data.head())

        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(data.describe())

        # Select forecasting model
        model_option = st.selectbox("Select Forecasting Model", ["Prophet", "ARIMA", "Exponential Smoothing"])

        # Input for forecast periods
        periods = st.slider("Select number of future periods to forecast", 1, 365, 30)

        # Forecasting button
        if st.button("Forecast"):
            if model_option == "Prophet":
                model = Prophet()
                model.fit(data)

                # Make future predictions
                future = model.make_future_dataframe(periods=periods)
                forecast = model.predict(future)

                # Show forecasted data
                st.subheader("Forecast Data")
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                # Plot the forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual', mode='lines'))
                fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast', mode='lines'))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_upper'], fill=None, mode='lines',
                    line=dict(color='lightgrey'), showlegend=False))
                fig.add_trace(go.Scatter(
                    x=forecast['ds'], y=forecast['yhat_lower'], fill='tonexty', mode='lines',
                    line=dict(color='lightgrey'), showlegend=False))
                st.subheader("Forecast Plot")
                st.plotly_chart(fig)

                # Calculate and display evaluation metrics
                forecast_df = forecast[['ds', 'yhat']].set_index('ds')
                merged_df = data.set_index('ds').join(forecast_df, how='left', rsuffix='_forecast')

                # Drop NaN values for actual vs. forecast comparison
                merged_df = merged_df.dropna(subset=['y', 'yhat'])
                mae = mean_absolute_error(merged_df['y'], merged_df['yhat'])
                mse = mean_squared_error(merged_df['y'], merged_df['yhat'])
                r2 = r2_score(merged_df['y'], merged_df['yhat'])

                st.subheader("Model Evaluation Metrics")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # Residuals plot
                residuals = merged_df['y'] - merged_df['yhat']
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=merged_df.index, y=residuals, mode='lines', name='Residuals'))
                st.subheader("Residuals Plot")
                st.plotly_chart(fig_residuals)

            elif model_option == "ARIMA":
                # ARIMA model requires the 'y' to be stationary
                order = st.selectbox("Select ARIMA Order (p, d, q)", [(1, 1, 1), (2, 1, 2), (1, 1, 2), (2, 1, 1)])
                model = ARIMA(data['y'], order=order)
                model_fit = model.fit()

                # Forecasting
                forecast = model_fit.forecast(steps=periods)
                forecast_dates = pd.date_range(start=data['ds'].max() + pd.Timedelta(days=1), periods=periods)

                # Show forecasted data
                st.subheader("Forecast Data")
                forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast})
                st.write(forecast_df)

                # Plot the forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual', mode='lines'))
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', mode='lines'))
                st.subheader("Forecast Plot")
                st.plotly_chart(fig)

                # Calculate and display evaluation metrics
                # Use the last known actual value for evaluation
                actual_last_value = data['y'].iloc[-len(forecast):].values
                mae = mean_absolute_error(actual_last_value, forecast)
                mse = mean_squared_error(actual_last_value, forecast)
                r2 = r2_score(actual_last_value, forecast)

                st.subheader("Model Evaluation Metrics")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # Residuals plot
                residuals = actual_last_value - forecast
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=forecast_dates, y=residuals, mode='lines', name='Residuals'))
                st.subheader("Residuals Plot")
                st.plotly_chart(fig_residuals)

            elif model_option == "Exponential Smoothing":
                model = ExponentialSmoothing(data['y'], trend='add', seasonal='add', seasonal_periods=7)
                model_fit = model.fit()

                # Forecasting
                forecast = model_fit.forecast(steps=periods)
                forecast_dates = pd.date_range(start=data['ds'].max() + pd.Timedelta(days=1), periods=periods)

                # Show forecasted data
                st.subheader("Forecast Data")
                forecast_df = pd.DataFrame({'ds': forecast_dates, 'yhat': forecast})
                st.write(forecast_df)

                # Plot the forecast
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=data['ds'], y=data['y'], name='Actual', mode='lines'))
                fig.add_trace(go.Scatter(x=forecast_df['ds'], y=forecast_df['yhat'], name='Forecast', mode='lines'))
                st.subheader("Forecast Plot")
                st.plotly_chart(fig)

                # Calculate and display evaluation metrics
                # Use the last known actual value for evaluation
                actual_last_value = data['y'].iloc[-len(forecast):].values
                mae = mean_absolute_error(actual_last_value, forecast)
                mse = mean_squared_error(actual_last_value, forecast)
                r2 = r2_score(actual_last_value, forecast)

                st.subheader("Model Evaluation Metrics")
                st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
                st.write(f"Mean Squared Error (MSE): {mse:.2f}")
                st.write(f"R-squared: {r2:.2f}")

                # Residuals plot
                residuals = actual_last_value - forecast
                fig_residuals = go.Figure()
                fig_residuals.add_trace(go.Scatter(x=forecast_dates, y=residuals, mode='lines', name='Residuals'))
                st.subheader("Residuals Plot")
                st.plotly_chart(fig_residuals)

    else:
        st.error("Make sure the CSV file contains 'ds' (date) and 'y' (numeric) columns.")

# Footer
st.write("Powered by Streamlit and Prophet, ARIMA, Exponential Smoothing")
