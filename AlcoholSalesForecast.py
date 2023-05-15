import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image
plt.style.use("dark_background")

# Load the data
data = pd.read_csv(r"C:\Users\91936\AlcoholSalesPrediction\AlcoholSales.csv")

# Convert DATE column to datetime
data['DATE'] = pd.to_datetime(data['DATE'])

# Prepare the features and target variables
X = data.index.values.reshape(-1, 1)  # Use the index as the feature
y = data['S4248SM144NCEN'].values

# Create and train the model
model = RandomForestRegressor()
model.fit(X, y)

# Define a function for sales forecasting
def forecast_sales(date):
    days_since = (pd.to_datetime(date) - data['DATE'].max()).days
    index = len(data) + days_since
    return model.predict([[index]])

# Streamlit app
st.title('Liquor Sales Forecast')
image = Image.open("D:\download\pexels-chris-f-1283219.jpg")
st.image(image)
show_dataset = st.checkbox('Show Original Dataset')
if show_dataset:
    st.subheader('Original Dataset')
    st.write(data)
st.write('Enter a date to forecast the sales:')

# User input
user_date = st.date_input('Date')

# Perform the forecast
if st.button('Forecast'):
    result = forecast_sales(user_date)
    st.write('Predicted Sales:', result)

    # Plot the sales data and forecast
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual sales
    ax.plot(data['DATE'], data['S4248SM144NCEN'], color='blue', label='Actual Sales')
    
    # Extend the x-axis range to include the forecasted period
    x_range = pd.date_range(start=data['DATE'].min(), end=user_date)
    
    # Predicted sales from the end of the actual data till the given date
    predicted_dates = pd.date_range(start=data['DATE'].max(), end=user_date)
    predicted_sales = [forecast_sales(date) for date in predicted_dates]
    
    # Plot forecasted sales
    ax.plot(predicted_dates, predicted_sales, color='orange', label='Forecasted Sales')
    
    # Highlight the sales on the given date
    ax.axvline(user_date, color='red', linestyle='--', alpha=0.5)
    ax.scatter(user_date, result, color='red', label='Predicted Sales')
    
    # Set the x-axis range and labels
    ax.set_xlim(data['DATE'].min(), user_date)
    ax.set_xlabel('Date')
    ax.set_ylabel('Sales')
    ax.set_title('Liquor Sales Forecast')
    
    # Show the legend
    ax.legend()
    
    # Checkbox to display original dataset
 
    
    # Show the plot
    st.pyplot(fig)
