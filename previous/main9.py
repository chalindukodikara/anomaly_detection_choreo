######################################################################
############# Airline Passengers Dataset - Exponential Smoothing #####
######################################################################
# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

import numpy as np
import pandas as pd
from pandas.tseries.offsets import DateOffset
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs
from sklearn.metrics import mean_squared_error
from math import sqrt
from pmdarima import auto_arima
from scipy.signal import savgol_filter
import arimaHelpers as arimaFunctions
# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# Import csv data
choreaData=pd.read_csv('echo_service_normal_variation_1_node1_merged_metrics.csv')
# We need to set the Month column as index and convert it into datetime
choreaData.set_index('datetime', inplace=True) #inplace = If True, modifies the DataFrame in place (do not create a new object)
choreaData = choreaData.drop(["in_throughput", "in_progress_requests", "http_error_count", 'ballerina_error_count', 'cpu', 'memory', 'cpuPercentage', 'memoryPercentage'], axis=1)
choreaData.index = pd.to_datetime(choreaData.index) #Convert argument to datetime.
choreaData.head()

# REMOVING 0 Rows
while True:
	x = 0
	y = len(choreaData)
	for i in range(len(choreaData)):
		if choreaData['in_avg_response_time'][i] == 0:
			choreaData = choreaData.drop(choreaData.head(len(choreaData)).index[i])
			break
		if i == y-1:
			x = 1
	if x == 1:
		break

# Drop first 2 rows since it contains abnormal data
choreaData = choreaData.drop(choreaData.head(len(choreaData)).index[0])
choreaData = choreaData.drop(choreaData.head(len(choreaData)).index[0])
# Split data set into train and test set
size = int(len(choreaData) * 0.7)
forecast_data, test = choreaData[0:size], choreaData[size:len(choreaData)]

print(forecast_data['in_avg_response_time'])
train_airline = forecast_data[:120]
test_airline = forecast_data[120:]
#
fitted_model = ExponentialSmoothing(train_airline['in_avg_response_time'],trend='mul',seasonal='mul',seasonal_periods=365).fit()
test_predictions = fitted_model.forecast(171)
print(test_predictions)
train_airline['in_avg_response_time'].plot(legend=True,label='TRAIN')
test_airline['in_avg_response_time'].plot(legend=True,label='TEST',figsize=(6,4))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()