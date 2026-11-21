# Ex.No: 6               HOLT WINTERS METHOD
#### Date: 21.11.2026
#### Name: Subhash V
#### Reg.No.: 212224240163



### AIM:
To implement the Holt Winters Method Model using Python.

### ALGORITHM:
1. You import the necessary libraries
2. You load a CSV file containing daily sales data into a DataFrame, parse the 'date' column as
datetime, and perform some initial data exploration
3. You group the data by date and resample it to a monthly frequency (beginning of the month
4. You plot the time series data
5. You import the necessary 'statsmodels' libraries for time series analysis
6. You decompose the time series data into its additive components and plot them:
7. You calculate the root mean squared error (RMSE) to evaluate the model's performance
8. You calculate the mean and standard deviation of the entire sales dataset, then fit a Holt-
Winters model to the entire dataset and make future predictions
9. You plot the original sales data and the predictions

### DATASET: 
Indian_Climate_Dataset_2024_2025.csv

### PROGRAM:
```PY
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load the dataset, perform data exploration
data = pd.read_csv('/content/Indian_Climate_Dataset_2024_2025.csv', parse_dates=['Date'])
data.head()

# Average the 10 cities for each date, giving one temperature per day
data_daily = data.groupby('Date')['Temperature_Avg (°C)'].mean()
data_daily = data_daily.resample('D').mean()
data_daily.head()
data_daily.plot()
plt.title('Daily Average Temperature (All Cities)')
plt.show()

# Scale the data and check for seasonality
scaler = MinMaxScaler()
scaled_data = pd.Series(scaler.fit_transform(data_daily.values.reshape(-1, 1)).flatten(),
                        index=data_daily.index)
scaled_data.plot()
plt.title('Scaled Data')
plt.show()

decomposition = seasonal_decompose(data_daily, model="additive", period=30)
decomposition.plot()
plt.show()

# Split test, train data, create a model using Holt-Winters method,
# train with train data and evaluate the predictions against test data
scaled_data = scaled_data + 1   # multiplicative seasonality can't handle non-positive values
train_data = scaled_data[:int(len(scaled_data) * 0.8)]
test_data = scaled_data[int(len(scaled_data) * 0.8):]

model_add = ExponentialSmoothing(train_data, trend='add', seasonal='mul', seasonal_periods=30).fit()
test_predictions_add = model_add.forecast(steps=len(test_data))

ax = train_data.plot()
test_predictions_add.plot(ax=ax)
test_data.plot(ax=ax)
ax.legend(["train_data", "test_predictions_add", "test_data"])
ax.set_title('Visual evaluation')
plt.show()

print("RMSE:", np.sqrt(mean_squared_error(test_data, test_predictions_add)))
print("MAE:", mean_absolute_error(test_data, test_predictions_add))
print("Std and Mean of scaled data:", np.sqrt(scaled_data.var()), scaled_data.mean())

# Create the final model, predict future data and plot it
final_model = ExponentialSmoothing(data_daily, trend='add', seasonal='mul', seasonal_periods=30).fit()
final_predictions = final_model.forecast(steps=int(len(data_daily) / 4))   # next ~6 months

ax = data_daily.plot()
final_predictions.plot(ax=ax)
ax.legend(["data_daily", "final_predictions"])
ax.set_xlabel('Date')
ax.set_ylabel('Average Temperature (°C)')
ax.set_title('Prediction')
plt.show()

print(final_predictions.head())
```
### OUTPUT:
*SCALED DATA PLOT*

<img width="547" height="471" alt="download" src="https://github.com/user-attachments/assets/67e04b84-a991-48f7-aee0-9dff8c952e8f" />

*DECOMPOSED PLOT*

<img width="630" height="470" alt="download" src="https://github.com/user-attachments/assets/62c28645-687b-4688-8ebc-93b012610d5a" />


*TEST_PREDICTION*

<img width="547" height="471" alt="download" src="https://github.com/user-attachments/assets/3b3b5686-83a0-4bc2-a074-99b98415b22d" />

*Model performance metrics:*

RMSE: 0.15865860324161365

*Standard deviation and mean:*

(0.16621672019180164,  1.4307865627572722)

*FINAL_PREDICTION*

<img width="571" height="471" alt="download" src="https://github.com/user-attachments/assets/59f4c0a8-dbc8-4e67-9cfa-afb790fc3fd2" />


### RESULT:
Thus the program run successfully based on the Holt Winters Method model.
