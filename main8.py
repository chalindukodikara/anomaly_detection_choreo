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
import holtWintersModel as holtWinters
import arimaModel


######################################################################
###################### DATA PREPARING ################################
######################################################################

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
choreaData['in_avg_response_time'] = savgol_filter(choreaData['in_avg_response_time'], 15, 8)
train_not_smoothed, test = choreaData[0:size], choreaData[size:len(choreaData)]

# Smooth dataset. window size 15, polynomial order 8
# train_not_smoothed = train_not_smoothed[150:]
# train_smoothed = savgol_filter(train_not_smoothed['in_avg_response_time'], 15, 8)

# Copy dataframe object into another object
train = train_not_smoothed.copy()

test = test[:30]
# train['in_avg_response_time'] = np.nan
# train['in_avg_response_time'] = train_smoothed
# Print smoothed data
# print(train)
# test['in_avg_response_time'] = savgol_filter(test['in_avg_response_time'], 15, 8)

print(train)
print('sssssssssssssss')
print(test)
# Plot original data
train_not_smoothed.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency', title = 'Training set and test set')
plt.show()

# Plot training data and test data with smoothed data
train_not_smoothed.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency', title = 'Training set and test set')
plt.plot(test.index, test, label='Test data')
plt.plot(train.index, train, label='Smoothed data' )
plt.legend(loc="upper left")
plt.show()

# Plot training data and smoothed data in small time frame
train_not_smoothed.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency',label = 'Latency', title = 'Smoothed data in small time frame', xlim=['2021-02-17 11:10','2021-02-17 11:40'])
# plt.plot(test.index, test)
plt.plot(train.index, train, label='Smoothed data')
plt.legend(loc="upper left")
plt.show()

# Print training and test set sizes
print('Training set size : ', len(train))
print('Test set size : ', len(test))

######################################################################
###################### ARIMA MODEL ###################################
######################################################################

arimaModel.main(train, test)

######################################################################
###################### HOLT WINTERS ##################################
######################################################################

# holtWinters.main(train, test)