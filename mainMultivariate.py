import numpy as np
from pandas.tseries.offsets import DateOffset
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import datetime
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
from pmdarima.arima.utils import ndiffs
from sklearn.metrics import mean_squared_error
from pmdarima import auto_arima
import holtWintersModel as holtWinters
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import arimaModel
from sklearn.preprocessing import StandardScaler
from math import sqrt
import os
# check prophet version
import fbprophet
import fbProphet
import thymeBoost
import varmaModel
import csv
from scipy.stats import boxcox
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.varmax import VARMAX
import numpy as np
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
from timeit import default_timer as timer
import arimaHelpers
import warnings
warnings.filterwarnings("ignore")

######################################################################
###################### VARIABLES #####################################
######################################################################
dataSetsList = ['echo_service_normal_variation_1_node1_merged_metrics.csv', 'echo_service_normal_variation_2_merged_metrics.csv', 'echo_service_normal_variation_3_merged_metrics.csv', 'echo_service_normal_variation_4_merged_metrics.csv',
				'echo_service_normal_variation_5_merged_metrics.csv', 'echo_service_normal_variation_6_merged_metrics.csv', 'echo_service_normal_variation_7_merged_metrics.csv', 'echo_service_normal_variation_8_merged_metrics.csv', 'echo_service_normal_variation_9_merged_metrics.csv',
				'service_chaining_normal_variation_1_node1_merged_metrics.csv', 'service_chaining_normal_variation_2_merged_metrics.csv', 'service_chaining_normal_variation_3_merged_metrics.csv', 'service_chaining_normal_variation_4a_merged_metrics.csv',
				'service_chaining_normal_variation_4b_merged_metrics.csv', 'service_chaining_normal_variation_5_merged_metrics.csv', 'service_chaining_normal_variation_6_merged_metrics.csv', 'service_chaining_normal_variation_7_node1_merged_metrics.csv',
				'service_chaining_normal_variation_7_node2_merged_metrics.csv', 'service_chaining_normal_variation_8_merged_metrics.csv', 'service_chaining_normal_variation_9_merged_metrics.csv', 'simple_passthrough_normal_variation_1_node1_merged_metrics.csv',
				'simple_passthrough_normal_variation_2_merged_metrics.csv', 'simple_passthrough_normal_variation_3_merged_metrics.csv', 'simple_passthrough_normal_variation_4_merged_metrics.csv', 'simple_passthrough_normal_variation_5_merged_metrics.csv',
				'simple_passthrough_normal_variation_6_merged_metrics.csv', 'simple_passthrough_normal_variation_7_merged_metrics.csv', 'simple_passthrough_normal_variation_8_merged_metrics.csv', 'simple_passthrough_normal_variation_9_merged_metrics.csv',
				'slow_backend_normal_variation_1_node1_merged_metrics.csv', 'slow_backend_normal_variation_2_merged_metrics.csv', 'slow_backend_normal_variation_3_merged_metrics.csv', 'slow_backend_normal_variation_4_merged_metrics.csv',
				'slow_backend_normal_variation_5_merged_metrics.csv', 'slow_backend_normal_variation_6_merged_metrics.csv', 'slow_backend_normal_variation_7_merged_metrics.csv', 'slow_backend_normal_variation_8_merged_metrics.csv', 'slow_backend_normal_variation_9_merged_metrics.csv']

anomalyDataSetsList = ['echo_service_cpu_hog_spikes_merged_metrics.csv', 'echo_service_cpu_hog_step_merged_metrics.csv', 'echo_service_user_surge_spike_node1_merged_metrics.csv', 'echo_service_user_surge_spike_node2_merged_metrics.csv',
					   'echo_service_user_surge_trend_node1_merged_metrics.csv', 'echo_service_user_surge_trend_node2_merged_metrics.csv', 'service_chaining_backend_failure_node1_merged_metrics.csv', 'service_chaining_backend_failure_node2_merged_metrics.csv',
					   'service_chaining_cpu_hog_spikes_merged_metrics.csv', 'service_chaining_cpu_hog_step_merged_metrics.csv', 'service_chaining_user_surge_spike_node1_merged_metrics.csv', 'service_chaining_user_surge_spike_node2_merged_metrics.csv',
					   'service_chaining_user_surge_trend_node1_merged_metrics.csv', 'service_chaining_user_surge_trend_node2_merged_metrics.csv', 'service_chaining_user_surge_trend_node3_merged_metrics.csv', 'simple_passthrough_cpu_hog_spikes_merged_metrics.csv',
					   'simple_passthrough_cpu_hog_step_merged_metrics.csv', 'simple_passthrough_user_surge_spike_node1_merged_metrics.csv', 'simple_passthrough_user_surge_spike_node2_merged_metrics.csv', 'simple_passthrough_user_surge_spike_node3_merged_metrics.csv',
					   'simple_passthrough_user_surge_spike_node4_merged_metrics.csv', 'simple_passthrough_user_surge_trend_node1_merged_metrics.csv', 'simple_passthrough_user_surge_trend_node2_merged_metrics.csv', 'slow_backend_cpu_hog_spikes_merged_metrics.csv',
					   'slow_backend_cpu_hog_step_merged_metrics.csv', 'slow_backend_long_delay_node1_merged_metrics.csv', 'slow_backend_long_delay_node2_merged_metrics.csv', 'slow_backend_short_delay_node1_merged_metrics.csv', 'slow_backend_short_delay_node2_merged_metrics.csv',
					   'slow_backend_user_surge_spike_node1_merged_metrics.csv', 'slow_backend_user_surge_spike_node2_merged_metrics.csv', 'slow_backend_user_surge_trend_node1_merged_metrics.csv', 'slow_backend_user_surge_trend_node2_merged_metrics.csv']
print(len(dataSetsList))
trainTestRatio = 0.7 # Split percentage
with open('Accuracy.txt', 'w') as f:
	f.write('\n')

fields = [' ']
with open(r'Anomaly.csv', 'w') as fd:
	writer = csv.writer(fd)
	writer.writerow(fields)
def invert_difference(orig_data, diff_data, interval):
	return [round(diff_data[i-interval] + orig_data[i-interval], 6) for i in range(interval, len(orig_data))]
######################################################################
###################### DATA PREPARING ################################
######################################################################
for dataSetID in [9, 20, 29]:
	csvFileName = dataSetsList[dataSetID]
	print(csvFileName)

	# Import csv data
	dataSet_v1 = pd.read_csv('normal_data/' + csvFileName)
	# We need to set the Month column as index and convert it into datetime
	dataSet_v1.set_index('datetime', inplace=True) # inplace = If True, modifies the DataFrame in place (do not create a new object)
	dataSet_v1 = dataSet_v1.drop(["in_progress_requests", "http_error_count", 'ballerina_error_count', 'cpuPercentage', 'memoryPercentage'], axis=1)
	dataSet_v1.index = pd.to_datetime(dataSet_v1.index) # Convert argument to datetime.
	dataSet_v1.head()

	# column name that is used
	columnList = ['in_avg_response_time', 'in_throughput', 'cpu', 'memory']
	# REMOVING 0 Rows
	for columnName in columnList:
		while True:
			x = 0
			y = len(dataSet_v1)
			for i in range(len(dataSet_v1)):
				if dataSet_v1[columnName][i] == 0:
					dataSet_v1 = dataSet_v1.drop(dataSet_v1.head(len(dataSet_v1)).index[i])
					break
				if i == y-1:
					x = 1
			if x == 1:
				break

	# dataSet_v1 = dataSet_v1.drop(["in_avg_response_time"], axis=1)

	# Drop first 4 rows since it contains abnormal data
	dataSet_v1 = dataSet_v1.drop(dataSet_v1.head(len(dataSet_v1)).index[0])
	dataSet_v1 = dataSet_v1.drop(dataSet_v1.head(len(dataSet_v1)).index[0])
	dataSet_v1 = dataSet_v1.drop(dataSet_v1.head(len(dataSet_v1)).index[0])
	dataSet_v1 = dataSet_v1.drop(dataSet_v1.head(len(dataSet_v1)).index[0])

	###################### Check Stationary ##############################
	######################################################################
	# Make the whole series stationary, all the columns will be stationary
	dataSet_v2 = dataSet_v1.copy()
	print(dataSet_v2.info())
	stationary = [False, False, False, False]
	numOfDifferences = 0
	# True = 1, False = 0. So sum gives a value.
	while sum(stationary) != 4:
		x = 0
		for name, column in dataSet_v2[columnList].iteritems():
			is_stationary = arimaHelpers.stationaryCheck(dataSet_v2[name], name)
			stationary[x] = is_stationary
			print(name, stationary)
			print('\n')
			x += 1

		if sum(stationary) != 4:
			numOfDifferences += 1
			dataSet_v2 = dataSet_v2.diff()
			dataSet_v2.dropna(inplace=True)

	print(dataSet_v2.info())
	print(dataSet_v2)

	###################### Check Correlation #############################
	######################################################################
	arimaHelpers.cointegration_test(dataSet_v1[columnList])
	print('\n')
	print('Correlation between each time series')
	print(dataSet_v1.corr())

	# # Power Transform
	# # dataSet_v1[column], lmbda = boxcox(dataSet_v1[column])
	#
	# # dataSet_v1['in_avg_response_time'] = np.log(dataSet_v1['in_avg_response_time'])
	# dataSet_v1.hist()
	# plt.show()
	# print(dataSet_v1.info())
	#
	# print(lowest, highest)
	#
	#
	######################################################################
	########################### Split Size ###############################
	######################################################################
	# Split data set into train and test set
	# size = int(len(dataSet_v1) * trainTestRatio)
	size = 50

	# ###################### Smoothing #####################################
	# ######################################################################
	# dataset_v1 = Original data, dataset_v2 = Differenced data to make stationary, dataset_v3 = Smoothed data
	# Savgol filter, window length, polynomial order : 15, 8
	dataSet_v3 = dataSet_v1.copy()
	for column in columnList:
		dataSet_v3[column] = savgol_filter(dataSet_v1[column], 15, 8)
	print(dataSet_v3)
	dataSet_v3 = dataSet_v3[:150]
	# Smoothed split
	train, test = dataSet_v3[0:size], dataSet_v3[size:len(dataSet_v3)]

	# Without smoothing
	train_v1, test_v1 = dataSet_v1[0:size], dataSet_v1[size:len(dataSet_v2)]

	# x = train.diff(periods = 2)
	# x.dropna(inplace=True)
	# print(train)
	# print(x)
	# for i in range(len(columnList)):
	# 	print(invert_difference(train[columnList[i]].values, x[columnList[i]].values, 2))

	# Plot original data
	dataSet_v3.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'y value', title = csvFileName + ' Tr set and test set')
	plt.savefig(str(dataSetID) + '.' + ' Train and Test Set (' + csvFileName + ').png', dpi=300, bbox_inches='tight')
	plt.show()

	for i in range(len(columnList)):
		dataSet_v3[columnList[i]].plot(figsize=(10, 7), xlabel='Time', ylabel= columnList[i],
						title=csvFileName + ' Tr set and test set' + columnList[i])
		plt.savefig(str(dataSetID) + '.' + ' Metric (' + columnList[i] + ').png', dpi=300,
					bbox_inches='tight')
		plt.show()

	for i in range(len(columnList)):
		# Plot training data and test data with smoothed data
		train_v1[columnList[i]].plot(figsize=(10, 7), xlabel = 'Time', ylabel = columnList[i], title = 'Original data and smoothed data '+ str(columnList[i]), xlim=[str(train.index[0])[:-9], str(train.index[-1])[:-9]])
		plt.plot(train.index, train[columnList[i]], label='Smoothed data' )
		plt.legend(loc="best")
		plt.savefig(str(dataSetID) + '.' + ' Smoothed and original data Metric (' + columnList[i] + ').png', dpi=300,
					bbox_inches='tight')
		plt.show()

	# # print(train, test)
	#
	# ######################################################################
	# ###################### ARIMA MODEL ###################################
	# ######################################################################
	# # arimaModel.main(train, test, dataSetID, column)

	varmaModel.main(train, test, dataSetID, columnList, train_v1, test_v1)

