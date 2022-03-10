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

def decomposeData(dataSet, dataColumn):
    # Decompose data into trends and seasonal
    decomposition = seasonal_decompose(dataColumn, model='multiplicative', period=1)

    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    plt.figure(figsize=(16, 7))
    plt.suptitle('View trend and seasonality')
    # fig = plt.figure(1)

    plt.subplot(411)
    plt.plot(dataSet, label='Original')
    plt.legend(loc='best')
    plt.subplot(412)
    plt.plot(trend, label='Trend')
    plt.legend(loc='best')
    plt.subplot(413)
    plt.plot(seasonal, label='Seasonality')
    plt.legend(loc='best')
    # plt.subplot(414)
    # plt.plot(residual, label='Residuals')
    # plt.legend(loc='best')
    plt.show()

def checkStationary(dataColumn):
    result = adfuller(dataColumn.dropna())
    print('ADF Statistic: %f' % result[0])
    print('p-value: %f' % result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print('\t%s: %.3f' % (key, value))
    if result[1] > 0.05:
        print("Dataset is not stationary")
    else:
        print("Dataset is stationary")

def dickeyFullerTest(train):
    ## Adf Test, Dickey–Fuller test
    print('adf test results : ', ndiffs(train, test='adf'))
    # KPSS test, Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test
    print('kpss test results : ', ndiffs(train, test='kpss'))
    # PP test, Phillips–Perron test is a unit root test
    print('pp test results : ', ndiffs(train, test='pp'))

def takeDifference(train, n):
    # Take first difference
    train_copy = train.copy()

    print('Take ' + str(n) + ' number of differences')
    if n == 1:
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'].plot(title='After first difference')
        checkStationary(train_copy['in_avg_response_time'])
        plt.show()

    if n == 2:
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'].plot(title='After second difference')
        checkStationary(train_copy['in_avg_response_time'])
        plt.show()

    if n == 3:
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'] = train_copy['in_avg_response_time'] - train_copy['in_avg_response_time'].shift(1)
        train_copy['in_avg_response_time'].plot(title='After second difference')
        checkStationary(train_copy['in_avg_response_time'])
        plt.show()

    plot_acf(train_copy['in_avg_response_time'].dropna(), title='ACF Diagram after ' + str(n) + ' number of differences')
    plot_pacf(train_copy['in_avg_response_time'].dropna(), title='PACF Diagram after ' + str(n) + ' number of differences')
    plt.show()

def forecastAccuracy(forecast, actual):
    mape = np.mean(np.abs(forecast - actual)/np.abs(actual))  # MAPE
    me = np.mean(forecast - actual)             # ME
    mae = np.mean(np.abs(forecast - actual))    # MAE
    mpe = np.mean((forecast - actual)/actual)   # MPE
    rmse = np.mean((forecast - actual)**2)**.5  # RMSE
    mse = mean_squared_error(forecast, actual)
    corr = np.corrcoef(forecast, actual)[0,1]   # corr
    mins = np.amin(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    maxs = np.amax(np.hstack([forecast[:,None],
                              actual[:,None]]), axis=1)
    # minmax = 1 - np.mean(mins/maxs)             # minmax



    accuracy = {'Mean Absolute Percentage Error (MAPE)':mape,'Root Mean Squared Error (RMSE)':rmse, 'Mean Absolute Error (MAE)':mae}
    with open('Accuracy.txt', 'a') as f:
        for metric in accuracy:
            line = metric + ': %.5f' % accuracy[metric]
            print(line)
            f.write(line)
            f.write('\n')
        f.write('\n')

def plotFinalResult(train, test, result):
    # Plot whole datasets including predictions
    plt.figure(figsize=(10, 7))
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.plot(train.index, train['in_avg_response_time'], label='Training set values')
    plt.plot(train.index, result.predict(1, len(train)), label='Model values for training set')
    plt.plot(test.index, test['in_avg_response_time'], label='Test set values')
    plt.plot(test.index, test['predicted'], label='Predicted fot test set')
    plt.fill_between(test.index, test['lower_latency'], test['higher_latency'], color='#ff7823', alpha=0.3,
                     label="confidence interval (95%)");
    plt.legend(loc='best')
    plt.show()

    # Plot test set with predictions
    plt.figure(figsize=(10, 7))
    plt.title('Test set values and predictions')
    plt.xlabel('Time')
    plt.ylabel('Latency')
    plt.plot(test.index, test['in_avg_response_time'], label='Test set values')
    plt.plot(test.index, test['predicted'], label='Predicted values')
    plt.fill_between(test.index, test['lower_latency'], test['higher_latency'], color='#ff7823', alpha=0.3,label="confidence interval (95%)");
    plt.legend(loc='best')
    plt.show()

    # test.plot(figsize=(10, 7), xlabel = 'Time', ylabel = 'Latency',label = 'Latency', title = 'ARIMA - Train data and fitted model in small time frame')
    # plt.fill_between(test.index, test['lower_latency'], test['higher_latency'], color='#ff7823', alpha=0.3, label="confidence interval (95%)");
    # plt.legend(loc='best')
    # plt.show()

def findAnomalies(predictionValues, testValues, threshold):
    squaredErrors = []
    # Add the squared errors between predictions and test values
    for i in range(len(testValues)):
        squaredErrors.append((predictionValues[i] - testValues[i]) ** 2)
    # print('squared errors inside', len(squared_errors), len(squaredErrors), squaredErrors) 568, 169
    anomaly = (squaredErrors >= threshold).astype(int)
    return anomaly

def plotAnomaly(train, test, result, pred):
    # Get residual errors
    squared_errors = result.resid.values ** 2
    # Define threshold
    threshold = np.mean(squared_errors) + 2 * np.std(squared_errors)
    # Find anomalies
    anomaly = findAnomalies(pred.predicted_mean.values.tolist(), test['in_avg_response_time'], threshold)

    test['Anomaly'] = anomaly
    testAnomaly = test.copy()

    # Check whether there are anomalies
    if np.isin(1, test['Anomaly']):
        # Get rows with anomalies and remove others
        while True:
            x = 0
            y = len(testAnomaly)
            for i in range(len(testAnomaly)):
                if testAnomaly['Anomaly'][i] == 0:
                    testAnomaly = testAnomaly.drop(testAnomaly.head(len(testAnomaly)).index[i])
                    break
                if i == y - 1: # at the end of the list
                    x = 1
            if x == 1:
                break

        print(testAnomaly)

        # Plot test set with predictions
        plt.figure(figsize=(10, 7))
        plt.title('Test set values and predictions')
        plt.xlabel('Time')
        plt.ylabel('Latency')
        plt.plot(test.index, test['in_avg_response_time'], label='Test set values')
        plt.plot(test.index, test['predicted'], label='Predicted values')
        # Plot anomalies
        plt.scatter(testAnomaly.index, testAnomaly['in_avg_response_time'], marker='x', color='red', label='Anomaly')
        plt.fill_between(test.index, test['lower_latency'], test['higher_latency'], color='#ff7823', alpha=0.3,label="confidence interval (95%)");
        plt.legend(loc='best')
        plt.show()

    else:
        print('There are no anomalies')