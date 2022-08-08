import numpy as np
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from datetime import datetime
from pmdarima import auto_arima
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
import arimaModel
from sklearn.preprocessing import StandardScaler
from math import sqrt
import os
import arimaFD
import arimaAnomalyDetection
import csv
from scipy.stats import boxcox

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

with open('Accuracy.txt', 'w') as f:
	f.write('\n')

fields = [' ']
with open(r'Anomaly.csv', 'w') as fd:
	writer = csv.writer(fd)
	writer.writerow(fields)

trainTestRatio = 0.7 # Split percentage
csvFiles = ['normal', 'anomaly']
x = 1
typeOfData = csvFiles[x]
columnsDrop = ["in_throughput", "in_progress_requests", "http_error_count", 'ballerina_error_count', 'cpu', 'memory', 'cpuPercentage', 'memoryPercentage']

normalList = [[0, 9, 20, 29],[9, 20]]
anomalyList = [[0, 10, 14, 16, 25, 31],[0, 10]]

if x == 0:
	if len(normalList[1]) != 0:
		dataFileLists = [normalList[x+1], anomalyList[x]]
	else:
		dataFileLists = [normalList[x], anomalyList[x]]
else:
	if len(anomalyList[1]) != 0:
		dataFileLists = [normalList[x-1], anomalyList[x]]
	else:
		dataFileLists = [normalList[x-1], anomalyList[x-1]]

splitSize = 40
######################################################################
###################### DATA PREPARING ################################
######################################################################
for dataSetID in dataFileLists[x]:
	# Check the data file type
	if typeOfData == 'anomaly':
		csvFileName = anomalyDataSetsList[dataSetID]
		print('CSV File Name :- ', csvFileName)
		# Import csv data
		dataSet_v1 = pd.read_csv('anomaly_data/' + csvFileName)
		columnsDrop.append('is_anomaly')
	else:
		csvFileName = dataSetsList[dataSetID]
		print(csvFileName)
		# Import csv data
		dataSet_v1 = pd.read_csv('normal_data/' + csvFileName)

	# We need to set the Month column as index and convert it into datetime
	dataSet_v1.set_index('datetime', inplace=True) # inplace = If True, modifies the DataFrame in place (do not create a new object)
	dataSet_v1 = dataSet_v1.drop(columnsDrop, axis=1)
	dataSet_v1.index = pd.to_datetime(dataSet_v1.index) # Convert argument to datetime.
	dataSet_v1.head()

	# column name that is used
	column = "in_avg_response_time"

	# REMOVING 0 Rows

	while True:
		x = 0
		y = len(dataSet_v1)
		for i in range(len(dataSet_v1)):
			if dataSet_v1["in_avg_response_time"][i] == 0:
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

	# xx = dataSet_v1.drop(labels = 'is_anomaly', axis = 1)
	# print(xx)
	###################### Take Difference ###############################
	######################################################################
	# # Difference
	# m = auto_arima(dataSet_v1[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
	# numOfDifferences = m.get_params()['order'][1]
	# x = difference(dataSet_v1[column].values, 1)
	# y = difference(dataSet_v1['in_throughput'].values, 1)
	# dataSet_v1 = dataSet_v1[1:]
	# dataSet_v1[column] = x
	# dataSet_v1['in_throughput'] = y
	# print(dataSet_v1)

	# dataSet_v1 = dataSet_v1.diff(periods=1)
	# dataSet_v1.dropna(inplace=True)  # Drop null values

	###################### Add WIP Feature ###############################
	######################################################################
	# # Add more features
	# dataSet_v1['wip'] = np.nan
	# dataSet_v1['wip'] = dataSet_v1['in_avg_response_time'] * dataSet_v1['in_throughput']
	# column = "wip"
	# dataSet_v1 = dataSet_v1.drop(
	# 	["in_avg_response_time",
	# 	 'in_throughput'], axis=1)
	# print(dataSet_v1)


	###################### Power Transform ###############################
	######################################################################
	# Power Transform
	# dataSet_v1[column], lmbda = boxcox(dataSet_v1[column])

	# dataSet_v1['in_avg_response_time'] = np.log(dataSet_v1['in_avg_response_time'])
	dataSet_v1.hist()
	plt.show()
	print('Dataset info : ')
	print(dataSet_v1.info())

	######################################################################
	###################### Get Anomaly Data ##############################
	######################################################################
	# csvFileNameAnomaly = 'echo_service_user_surge_spike_node2_merged_metrics.csv'
	#
	# # Import csv data
	# anomalyDataSet_v1 = pd.read_csv('anomaly_data/' + csvFileNameAnomaly)
	# # We need to set the Month column as index and convert it into datetime
	# anomalyDataSet_v1.set_index('datetime', inplace=True)  # inplace = If True, modifies the DataFrame in place (do not create a new object)
	# anomalyDataSet_v1 = anomalyDataSet_v1.drop(
	# 	["in_throughput", "in_progress_requests", "http_error_count", 'ballerina_error_count', 'cpu', 'memory',
	# 	 'cpuPercentage', 'memoryPercentage', 'is_anomaly'], axis=1)
	# anomalyDataSet_v1.index = pd.to_datetime(anomalyDataSet_v1.index)  # Convert argument to datetime.
	# anomalyDataSet_v1.head()
	#
	# # column name that is used
	# column = "in_avg_response_time"
	#
	# # REMOVING 0 Rows
	# while True:
	# 	x = 0
	# 	y = len(anomalyDataSet_v1)
	# 	for i in range(len(anomalyDataSet_v1)):
	# 		if anomalyDataSet_v1["in_avg_response_time"][i] == 0:
	# 			anomalyDataSet_v1 = anomalyDataSet_v1.drop(anomalyDataSet_v1.head(len(anomalyDataSet_v1)).index[i])
	# 			break
	# 		if i == y - 1:
	# 			x = 1
	# 	if x == 1:
	# 		break
	#
	# # anomalyDataSet_v1 = anomalyDataSet_v1.drop(["in_avg_response_time"], axis=1)
	#
	# # Drop first 4 rows since it contains abnormal data
	# anomalyDataSet_v1 = anomalyDataSet_v1.drop(anomalyDataSet_v1.head(len(anomalyDataSet_v1)).index[0])
	# anomalyDataSet_v1 = anomalyDataSet_v1.drop(anomalyDataSet_v1.head(len(anomalyDataSet_v1)).index[0])
	# anomalyDataSet_v1 = anomalyDataSet_v1.drop(anomalyDataSet_v1.head(len(anomalyDataSet_v1)).index[0])
	# anomalyDataSet_v1 = anomalyDataSet_v1.drop(anomalyDataSet_v1.head(len(anomalyDataSet_v1)).index[0])
	# # anomalyDataSet_v1 = anomalyDataSet_v1[:43]
	#
	# for i in range(1, len(anomalyDataSet_v1)):
	# 	idx = dataSet_v1.tail(1).index[0] + pd.Timedelta(seconds=10)
	# 	dataSet_v1.loc[idx] = round(anomalyDataSet_v1.iloc[i, 0], 6)

	########################### Split Size ###############################
	######################################################################
	# Split data set into train and test set
	# size = int(len(dataSet_v1) * trainTestRatio)
	# size = len(dataSet_v1)
	size = splitSize

	## Limit dataset size
	# dataSet_v1 = dataSet_v1[:int(len(dataSet_v1)*0.5)]


	###################### Standardization ###############################
	######################################################################
	# Standardize time series data
	# prepare data for standardization
	values = dataSet_v1.values
	values = values.reshape((len(values), 1))

	# train the standardization
	scaler = StandardScaler()
	scaler = scaler.fit(values)
	print('Mean: %f, StandardDeviation: %f' % (scaler.mean_, sqrt(scaler.var_)))

	# standardization the dataset and print the first 5 rows
	normalized = scaler.transform(values)
	# for i in range(5):
	# 	print(normalized[i])

	# inverse transform and print the first 5 rows
	# inversed = scaler.inverse_transform(normalized)
	# for i in range(5):
	# 	print(inversed[i])

	# dataSet_v1[column] = normalized/10
	print(dataSet_v1)
	dataSet_v1.hist()
	plt.show()

	########################### Smoothing ################################
	######################################################################
	# Savgol filter, window length, polynomial order : 15, 8
	dataSet_v2 = dataSet_v1.copy()
	dataSet_v2[column] = savgol_filter(dataSet_v1[column], 15, 8)
	print('Dataset after smoothing')
	print(dataSet_v2)

	# Smoothed split
	train, test = dataSet_v2[0:size], dataSet_v2[size:len(dataSet_v2)]

	# Without smoothing
	train_v1, test_v1 = dataSet_v1[0:size], dataSet_v1[size:len(dataSet_v2)] # Without savlog filter

	################## Find Lowest and Highest values ####################
	######################################################################
	# # Find lowest and highest value in the normal dataset
	# lowest = 9999999
	# highest = -9999999
	# while True:
	# 	x = 0
	# 	y = len(train)
	# 	for i in range(len(train)):
	# 		if train[column][i] < lowest:
	# 			lowest = train[column][i]
	# 		# print('lowest: ', lowest)
	# 		# print(dataSet_v1.iloc[i])
	# 		if train[column][i] > highest:
	# 			highest = train[column][i]
	# 		# print('highest: ', highest)
	# 		# print(dataSet_v1.iloc[i])
	# 		if i == y - 1:
	# 			x = 1
	# 	if x == 1:
	# 		break
	# lowHighDifference = round((highest - lowest), 5)
	# print('lowest and highest values ', lowest, highest)
	########################### Plot data ################################
	######################################################################

	# Plot original data
	dataSet_v1.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency(ms)', title = csvFileName + ' Tr set and test set')
	plt.savefig("plots_result/" + str(dataSetID) + '.' + ' Train and Test Set (' + csvFileName + ').png', dpi=300, bbox_inches='tight')
	plt.show()

	# Plot training data and test data with smoothed data
	train_v1.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency(ms)', title = 'Training set and test set')
	plt.plot(test_v1.index, test_v1, label='Test data')
	plt.plot(train.index, train[column], label='Smoothed data' )
	plt.legend(loc="upper left")
	plt.show()

	# # Plot training data and smoothed data in small time frame
	# train_v1.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency(ms)',label = 'Latency', title = 'Smoothed data in small time frame', xlim=[str(train.index[-20])[:-9], str(test.index[1])[:-9]])
	# # plt.plot(test.index, test)
	# plt.plot(train.index, train[column], label='Smoothed data')
	# plt.legend(loc="upper left")
	# plt.savefig(str(dataSetID) + '.' + ' Smoothed Data (' + str(1) + ').png', dpi=300, bbox_inches='tight')
	# plt.show()

	# # Plot training data and smoothed data in small time frame
	# train_v1.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency(ms)', label='Latency',
	# 			  title='Smoothed data in small time frame', xlim=[str(train.index[-40])[:-9], str(test.index[1])[:-9]])
	# # plt.plot(test.index, test)
	# plt.plot(train.index, train[column], label='Smoothed data')
	# plt.legend(loc="upper left")
	# plt.savefig(str(dataSetID) + '.' + ' Smoothed Data (' + str(2) + ').png', dpi=300, bbox_inches='tight')
	# plt.show()

	# # Plot training data and smoothed data in small time frame
	# test_v1.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency(ms)', label='Latency',
	# 			  title='Smoothed data in small time frame', xlim=[str(test.index[0])[:-9], str(test.index[-1])[:-9]])
	# # plt.plot(test.index, test)
	# plt.plot(test.index, test[column], label='Smoothed data')
	# plt.legend(loc="upper left")
	# plt.savefig(str(dataSetID) + '.' + ' Smoothed Data (' + str(2) + ').png', dpi=300, bbox_inches='tight')
	# plt.show()

	# Print training and test set sizes
	print('\n')
	print(str(dataSetID + 1) + '. Dataset Sizes ' + csvFileName)
	print('Training set size : ', len(train))
	print('Test set size : ', len(test))
	print('Dataset size : ', len(dataSet_v1))
	print('\n')
	print(1%5)
	######################################################################
	###################### ARIMA MODEL ###################################
	######################################################################
	# arimaModel.main(train, test, dataSetID, column)

	arimaAnomalyDetection.main(train, test, dataSetID, column, train_v1, test_v1)

	# arimaFD.main(train, test, dataSetID, column, lowHighDifference, train_v1, test_v1)
	######################################################################
	###################### HOLT WINTERS ##################################
	######################################################################

	# holtWinters.main(train, test, dataSetID, column)

	######################################################################
	######################## FB PROPHET ##################################
	######################################################################

	# fbProphet.main(dataSet_v2, dataSetID, size, column)


	######################################################################
	######################## THYME BOOST #################################
	######################################################################

	# thymeBoost.main(train, test, dataSetID, column)