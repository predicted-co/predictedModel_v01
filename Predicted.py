import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Load the datasets
train_data = pd.read_csv('data/raw/pizzasales.csv')
test_data = pd.read_csv('data/train/requirements.txt')

# Review the first row of the training data
print(train_data.head(1))

# Check the data types
print(train_data.dtypes)

# Fill in missing values in the discount_percent column
train_data['discount_percent'].fillna(0, inplace=True)

# Convert the date column into date format
train_data['date'] = pd.to_datetime(train_data['date'], format='%d-%b-%y')
test_data['date'] = pd.to_datetime(test_data['date'], format='%d-%b-%y')

# Train the machine learning model
X = train_data[['baseprice_USD', 'discount_percent', 'is_weekend', 'is_friday', 'is_holiday']]
y = train_data['sales_quantity']
X = sm.add_constant(X)  # Adds a constant term to the predictor
model = sm.OLS(y, X).fit()

# Find correlation between the discount and sales quantity on the full training data
correlation_full = train_data['discount_percent'].corr(train_data['sales_quantity'])
print(f'Correlation (full data): {correlation_full}')

# Find the same correlation for different subsets of restaurants and food items
correlation_by_restaurant_item = train_data.groupby(['restaurant', 'item_name']).apply(
    lambda x: x['discount_percent'].corr(x['sales_quantity'])
)
print(correlation_by_restaurant_item)

# Save the highest correlation pair
highest_correlation_pair = correlation_by_restaurant_item.idxmax()
highest_correlation_value = correlation_by_restaurant_item.max()
print(f'Highest correlation pair: {highest_correlation_pair} with value: {highest_correlation_value}')

# Plot time series data for each restaurant-item pair
for (restaurant, item), group in train_data.groupby(['restaurant', 'item_name']):
    plt.figure(figsize=(10, 5))
    plt.plot(group['date'], group['sales_quantity'], marker='o')
    plt.title(f'Time Series for {restaurant} - {item}')
    plt.xlabel('Date')
    plt.ylabel('Sales Quantity')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

# Build a regression model using the full training data
model_summary = model.summary()
print(model_summary)

# Try out different subsets of the data or predictors to get a better adjusted r-squared score
best_adjusted_r_squared = model.rsquared_adj
print(f'Best Adjusted R-squared: {best_adjusted_r_squared}')

# Store your best adjusted score
best_adjusted_score = best_adjusted_r_squared

# Subset the test data to match the training set
test_data_subset = test_data[['baseprice_USD', 'discount_percent', 'is_weekend', 'is_friday', 'is_holiday']]
test_data_subset['discount_percent'].fillna(0, inplace=True)
test_data_subset['date'] = pd.to_datetime(test_data_subset['date'], format='%d-%b-%y')

# Generate sales predictions using the test set
X_test = sm.add_constant(test_data_subset)
predictions = model.predict(X_test)

# Capture residuals of the data
residuals = test_data['sales_quantity'] - predictions

# Calculate the RMSE of the residuals
rmse = np.sqrt(np.mean(residuals**2))
print(f'RMSE of the residuals: {rmse}')
# End Generation Here
