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
import time

######################################################################
###################### HOLT WINTERS ##################################
######################################################################

def main(train, test, dataSetID, column):
    numOfSteps = 3
    immediate1stPredictions = []
    immediate2ndPredictions = []
    immediate3rdPredictions = []
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)

    models = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions)

    # Create 3 dataframes for 3 observations
    test1 = test.copy()
    test2 = test.copy()
    test3 = test.copy()

    # Remove 1st value from test1 set and remove first 2 values from test3, since immediate2ndPredictions doesnt contain 1st value of test set and immediate3rdPredictions doesnt contain 1st 2 values of test set
    test2 = test2.drop(test2.index[0])
    test3 = test3.drop(test3.index[[0, 1]])

    # Remove extra predictions
    immediate2ndPredictions = immediate2ndPredictions[:-1]
    immediate3rdPredictions = immediate3rdPredictions[:-2]

    # Add observation values into each test set
    test1['predictions'] = immediate1stPredictions
    test2['predictions'] = immediate2ndPredictions
    test3['predictions'] = immediate3rdPredictions

    pd.set_option("display.max_rows", None, "display.max_columns", None)

    print(test1)
    print(test2)
    print(test3)

    plotResults(dataSet, train, test1, models[0], 'Holt Winters - Immediate 1st Predictions',
                'Holt Winters - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
                'Holt Winters - Immediate 2nd Predictions', 'Holt Winters - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
                'Holt Winters - Immediate 3rd Predictions', 'Holt Winters - Immediate 3rd Predictions in small time frame',
                str(test1.index[1])[:-6], dataSetID)

    print(str(dataSetID + 1) + '. Accuracy metrics')
    # Get Accuracy values for fitted model and training set
    print('Fitted model and training set')
    arimaFunctions.forecastAccuracy(np.array(models[0].fittedvalues.values.tolist()), np.array(train[column]))

    # Get Accuracy values for immediate 1st values and test set
    print('Immediate 1st values and test set')
    arimaFunctions.forecastAccuracy(np.array(test1['predictions']), np.array(test1[column]))

    # Get Accuracy values for immediate 2nd values and test set
    print('Immediate 2nd values and test set')
    arimaFunctions.forecastAccuracy(np.array(test2['predictions']), np.array(test2[column]))

    # Get Accuracy values for immediate 3rd values and test set
    print('Immediate 3rd values and test set')
    arimaFunctions.forecastAccuracy(np.array(test3['predictions']), np.array(test3[column]))


def trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions):
    train1 = train.copy()
    train1['HWES2_MUL'] = ExponentialSmoothing(train1[column], trend='add').fit().fittedvalues
    print(train1)
    train1[[column, 'HWES2_MUL']].plot(figsize=(10, 7), title='Holt Winters Double Exponential Smoothing: Multiplicative Trend');
    plt.show()

    # Plot train data and fitted model in small time frame
    train1[[column, 'HWES2_MUL']].plot(figsize=(10, 7), title='Holt Winters Double Exponential Smoothing: Multiplicative Trend in small time frame', xlim=['2021-02-17 12:10', '2021-02-17 12:21']);
    plt.show()

    models = []
    for i in range(len(test)):
        # Create the model and fit it
        t1 = time.time()
        model = ExponentialSmoothing(train1[column], trend='add')
        result = model.fit()
        if i == 0 or i == 1 or i == 2:
            models.append(result)
        # Print summary of the model
        # print(result.summary())
        pred = result.forecast(3)
        pred = pred.to_frame() # Convert series into frame
        # print(pred.values[0][0], pred.values[1][0],type(pred.values))
        immediate1stPredictions.append(pred.values[0][0])
        immediate2ndPredictions.append(pred.values[1][0])
        immediate3rdPredictions.append(pred.values[2][0])
        # train.concat(test.iloc[0])
        train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
    return models

def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations
    startDate = '2021-02-17 12:18'
    endDate = '2021-02-17 12:40'

    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency', title=title1, xlim=[str(trainSet.index[-8])[:-9], str(testSet.index[int((len(testSet)/3)*2)])[:-9]])
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title1 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot whole dataset, fitted values and immediate observations in small time frame
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-6])[:-9], str(testSet.index[int(len(testSet)/3)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    #
    # Plot whole dataset, fitted values and immediate observations in small time frame
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-5])[:-9], str(testSet.index[int(len(testSet)/6)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()

# train['HWES2_ADD'] = ExponentialSmoothing(train[column],trend='add').fit().fittedvalues
    # train['HWES2_MUL'] = ExponentialSmoothing(train[column], trend='mul').fit().fittedvalues
    # train[[column, 'HWES2_MUL']].plot(figsize=(10, 7),
    #                                                   title='Holt Winters Double Exponential Smoothing: Multiplicative Trend');
    # plt.show()
    #
    # # Plot train data and fitted model in small time frame
    # train[[column, 'HWES2_MUL']].plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #                                                   title='Holt Winters - Train data and fitted model in small time frame',
    #                                                   xlim=['2021-02-17 11:10', '2021-02-17 11:40'])
    # plt.show()
    #
    # # Plot final results for ARIMA
    # fitted_model = ExponentialSmoothing(train[column], trend='add').fit()
    # test_predictions = fitted_model.forecast(171)
    # test_predictions = test_predictions.to_frame()
    # test['Holt Predictions'] = test_predictions.values
    # test[column].plot(legend=True, label='TEST', figsize=(10, 7))
    # plt.plot(test.index, test['Holt Predictions'], label='PREDICTION')
    # plt.legend(loc='best')
    # plt.title('Train, Test and Predicted Test using Holt Winters')
    # plt.show()
    # print(test)
    #
    # # Plot test set with predictions
    # plt.figure(figsize=(10, 7))
    # plt.xlabel('Time')
    # plt.ylabel('Latency')
    # plt.plot(train.index, train[column], label='Training set values')
    # plt.plot(train.index, train['HWES2_MUL'], label='Model values for training set')
    # plt.plot(test.index, test[column], label='Test set values')
    # plt.plot(test.index, test['Holt Predictions'], label='Predicted fot test set')
    # plt.legend(loc='best')
    # plt.show()