import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px

# Load the pizza sales data
pizza_file_path = 'data/raw/pizzasales.csv'
pizza_data = pd.read_csv(pizza_file_path)

# Convert order_date to datetime
pizza_data['order_date'] = pd.to_datetime(pizza_data['order_date'])

# Extract day of the week and hour from order_date
pizza_data['day_of_week'] = pizza_data['order_date'].dt.day_name()
pizza_data['hour'] = pizza_data['order_date'].dt.hour

# Create a strip plot using Plotly
fig = px.strip(pizza_data, x='order_date', y='hour', color='order_date')
fig.show()
    