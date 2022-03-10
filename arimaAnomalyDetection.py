import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import arimaHelpers as arimaFunctions
import csv

######################################################################
###################### ARIMA MODEL ###################################
######################################################################

def main(train, test, dataSetID, column, lowHighDifference, train_v1, test_v1):

    numOfSteps = 3
    # test = test[:8]
    immediate1stPredictions = []
    immediate2ndPredictions = []
    immediate3rdPredictions = []
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)
    dataSetPred = dataSet.copy()
    print('Low high difference : ', lowHighDifference)
    print('Training set size : ', len(train))
    print('Test set size : ', len(test))
    print('Dataset size : ', len(dataSet))

    # Train the model
    # immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions. immediate2ndPredictions, immediate3rdPredictions)
    testSetIsAnomaly, models = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column, lowHighDifference, test_v1)

    dataSetPred['predictions'] = np.nan
    for i in range(len(immediate1stPredictions)):
        dataSetPred.loc[dataSetPred.head(len(dataSetPred)).index[i + len(train)], 'predictions'] = immediate1stPredictions[i]



    pd.set_option("display.max_rows", None, "display.max_columns", 9)

    print(dataSet)
    print(testSetIsAnomaly)
    fields = [str(dataSetID + 1) + '. Accuracy metrics']
    with open(r'Anomaly.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow(fields)

    testSetIsAnomaly.to_csv('Anomaly.csv', mode='a')

    with open(r'Anomaly.csv', 'a') as fd:
        writer = csv.writer(fd)
        writer.writerow('\n')

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


    plotResults(dataSet, train, test1, models[0], 'ARIMA - Immediate 1st Predictions', 'ARIMA - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
                'ARIMA - Immediate 2nd Predictions', 'ARIMA - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
                'ARIMA - Immediate 3rd Predictions', 'ARIMA - Immediate 3rd Predictions in small time frame',
                str(test1.index[1])[:-6], dataSetID)

    # Plot final plots
    plotAnomaly(testSetIsAnomaly, dataSet, train, dataSetID, test_v1, column)
    # # plot the residuals of the model. The residuals are the difference between the original values and the predicted values from the model.
    # result.resid.plot(kind='kde', title='residuals of the model')
    # plt.show()

    print(str(dataSetID + 1) + '. Accuracy metrics')
    with open('Accuracy.txt', 'a') as f:
        f.write(str(dataSetID + 1) + '. Accuracy metrics')
        f.write('\n')
    # Get Accuracy values for fitted model and training set
    print('Fitted model and training set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Fitted model and training set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(models[0].fittedvalues.values.tolist()), np.array(train[column]))

    # Get Accuracy values for immediate 1st values and test set
    print('Immediate 1st values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 1st values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test1['predictions']), np.array(test1[column]))

    # Get Accuracy values for immediate 2nd values and test set
    print('Immediate 2nd values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 2nd values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test2['predictions']), np.array(test2[column]))

    # Get Accuracy values for immediate 3rd values and test set
    print('Immediate 3rd values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 3rd values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test3['predictions']), np.array(test3[column]))


def trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column, lowHighDifference, test_v1):
    # Find the suitable model using auto arima
    m = auto_arima(train[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
    print(m.summary())
    autoRegressive = m.get_params()['order'][0]
    numOfDifferences = m.get_params()['order'][1]
    movingAverage = m.get_params()['order'][2]

    # Plot train data and fitted model
    model = ARIMA(train[column], order=(autoRegressive, numOfDifferences, movingAverage))
    result = model.fit()

    train.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
               title='ARIMA - Train data and fitted model')
    plt.plot(train.index, result.predict(1, len(train)), label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + 'Train data and fitted model' + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot train data and fitted model in small time frame
    # train.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #              title='ARIMA - Train data and fitted model in small time frame',
    #              xlim=['2021-02-17 12:10', '2021-02-17 12:21'])
    # plt.plot(train.index, result.predict(1, len(train)), label='Fitted model')
    # plt.legend(loc='best')
    # plt.show()

    models = []
    train1 = train.copy() # Create new train dataframe to avoid changes in initial training dataset
    test1 = test.copy()
    test1['predictions'] = np.nan
    test1['is_anomaly'] = np.nan

    # Predict immediate 3 values each time and add them into 3 arrays
    for i in range(0, len(test)):
        if ((i+1)%20 == 0):
            m = auto_arima(train1[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
            print(m.summary())
            autoRegressive = m.get_params()['order'][0]
            numOfDifferences = m.get_params()['order'][1]
            movingAverage = m.get_params()['order'][2]
        model = ARIMA(train1[column], order=(autoRegressive, numOfDifferences, movingAverage))
        result = model.fit()
        if i == 0 or i == 1 or i == 2:
            models.append(result)

        pred = result.get_forecast(steps=numOfSteps)
        immediate1stPredictions.append(pred.predicted_mean.values[0])
        immediate2ndPredictions.append(pred.predicted_mean.values[1])
        immediate3rdPredictions.append(pred.predicted_mean.values[2])

        squared_errors = (result.resid.values) ** 2
        threshold = np.mean(squared_errors) + 3 * np.std(squared_errors) + lowHighDifference/2
        # print(threshold, np.mean(squared_errors), 3 * np.std(squared_errors), lowHighDifference/2,lowHighDifference)
        # print('threshold', threshold, np.mean(squared_errors), 3 * np.std(squared_errors), lowHighDifference)
        # ci = result.conf_int(0.05)
        # print(ci, ci[0], type(ci))
        # uncertainty = abs(ci[0, 0] - ci[0, 1])

        # Integer value is returned, not string
        anomaly = detectAnomaly(pred.predicted_mean.values[0], i, test, threshold, test_v1)

        # train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)

        if anomaly == 1:
            x = test.copy()
            x = x[i:i + 1]
            x.iloc[0, 0] = pred.predicted_mean.values[0]
            train1 = pd.concat([train1, x], ignore_index=False, axis=0)
        else:
            train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)

        test1.iloc[i, 1] = pred.predicted_mean.values[0]
        test1.iloc[i, 2] = str(anomaly)
    return test1, models

def detectAnomaly(predictedValue, i, test, threshold, test_v1):
    squaredError = (predictedValue - test.iloc[i, 0]) ** 2
    squaredError_1 = (predictedValue - test_v1.iloc[i, 0]) ** 2
    # print('Squared error, threshold', squaredError,squaredError_1, threshold)
    # if squaredError >= threshold:
    #     if squaredError_1 >= threshold:
    #         anomaly = 1
    #     else:
    #         anomaly = 0
    # else:
    #     anomaly = 0

    # if squaredError >= threshold or squaredError_1 >= threshold:
    #     anomaly = 1
    # else:
    #     anomaly = 0
    anomaly = (squaredError >= threshold).astype(int)
    # if anomaly == 1:
    #     print(test.iloc[i], test_v1.iloc[i])
    return anomaly

def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations
    startDate = '2021-02-17 12:18'
    endDate = '2021-02-17 12:40'

    # index = -13
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency', title=title1, xlim=[str(trainSet.index[-3])[:-9], str(testSet.index[int((len(testSet)/6)*5)])[:-9]])
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title1 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot whole dataset, fitted values and immediate observations in small time frame, index -11
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-3])[:-9], str(testSet.index[int(len(testSet)/3)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    #
    # Plot whole dataset, fitted values and immediate observations in small time frame, -9
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-3])[:-9], str(testSet.index[int(len(testSet)/6)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()

def plotAnomaly(test, dataSet, train, dataSetID, test_v1, column):
    testAnomaly = test.copy()
    exists = '1' in testAnomaly.values
    print('exists', exists)
    if (exists == True):
        while True:
            x = 0
            y = len(testAnomaly)
            for i in range(y):
                if testAnomaly['is_anomaly'][i] == '0':
                    testAnomaly = testAnomaly.drop(testAnomaly.head(len(testAnomaly)).index[i])
                    break
                if i == y - 1:
                    x = 1
            if x == 1:
                break
        print(testAnomaly)
    else:
        testAnomaly[column] = np.nan
        print('There are no anomalies in the dataset')

    printSet = dataSet.copy()
    # printSet = printSet.loc[str(train.index[-13])[:-9]:str(test.index[int((len(test) - 5))])[:-9]]
    printSet = printSet.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) - 5))])[:-9]]
    columnValues = printSet[column]
    # if column.max() < 2:
    #     maxValue = round(column.max() + 0.2 * abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), 3)
    #     minValue = round(column.min() - 0.2 * abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), 3)
    # else:
    #     maxValue = round(column.max() + 1.5 * abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), 3)
    #     minValue = round(column.min() - 1.5 * abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), 3)

    # Increase some amount of Y limit
    if columnValues.max() > 1:
        constantHigh = 1.5
    else:
        constantHigh = 0.5
    if abs(columnValues.min()) > 1:
        constantLow = 2
    else:
        constantLow = 0.3
    if columnValues.max() < 2:
        maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)
        minValue = round(columnValues.min() - 0.5 * constantLow, 3)
    else:
        maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
        minValue = round(columnValues.min() - 1.5 * constantLow, 3)

    # print(maxValue, minValue, column.max(), column.min())
    # dataSet.plot(figsize=(10, 7), title='Anomaly whole test set 1', xlabel='Time', ylabel='Latency',
    #              ylim=[minValue, maxValue],
    #              xlim=[str(train.index[-13])[:-9], str(test.index[int((len(test) - 1))])[:-9]])
    dataSet.plot(figsize=(10, 7), title='Anomaly whole test set 1', xlabel='Time', ylabel='Latency',
                 ylim=[minValue, maxValue],
                 xlim=[str(train.index[-3])[:-9], str(test.index[int((len(test) - 1))])[:-9]])
    plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')

    # plt.plot(test_v1.index, test_v1['in_avg_response_time'], color='green', label='Latency WO Smoothing')
    plt.plot(test.index, test['predictions'], color='orange', label='Predictions')
    plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '.' + ' Anomaly whole test set 1.png', dpi=300, bbox_inches='tight')
    plt.show()

    printSet = dataSet.copy()
    # printSet = printSet.loc[str(train.index[-13])[:-9]:str(test.index[int((len(test) / 3) * 1)])[:-9]]
    printSet = printSet.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) / 3) * 1)])[:-9]]
    columnValues = printSet[column]

    if columnValues.max() > 1:
        constantHigh = 1.5
    else:
        constantHigh = 0.5
    if abs(columnValues.min()) > 1:
        constantLow = 2
    else:
        constantLow = 0.3
    if columnValues.max() < 2:
        maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)
        minValue = round(columnValues.min() - 0.5 * constantLow, 3)
    else:
        maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
        minValue = round(columnValues.min() - 1.5 * constantLow, 3)

    # print(maxValue, minValue, column.max(), column.min())
    # dataSet.plot(figsize=(10, 7), title='Predictions for test set values 2', xlabel='Time', ylabel='Latency',
    #              ylim=[minValue, maxValue],
    #              xlim=[str(train.index[-13])[:-9], str(test.index[int((len(test) / 3) * 1)])[:-9]])
    dataSet.plot(figsize=(10, 7), title='Predictions for test set values 2', xlabel='Time', ylabel='Latency',
                 ylim=[minValue, maxValue],
                 xlim=[str(train.index[-3])[:-9], str(test.index[int((len(test) / 3) * 1)])[:-9]])
    plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')
    # plt.plot(test_v1.index, test_v1['in_avg_response_time'], color='green', label='Latency WO Smoothing')
    plt.plot(test.index, test['predictions'], color='orange', label='Predictions')
    plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '.' + ' Anomaly whole test set 2.png', dpi=300, bbox_inches='tight')
    plt.show()

    for i in range(int(len(test) / 30)):
        printSet = dataSet.copy()
        printSet = printSet.loc[str(test.index[i * 30])[:-9]:str(test.index[i * 30 + 30])[:-9]]
        columnValues = printSet[column]
        # print(printSet)
        if columnValues.max() > 1:
            constantHigh = 1.5
        else:
            constantHigh = 0.5
        if abs(columnValues.min()) > 1:
            constantLow = 2
        else:
            constantLow = 0.3
        if columnValues.max() < 2:
            maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)
            minValue = round(columnValues.min() - 0.5 * constantLow, 3)
        else:
            maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
            minValue = round(columnValues.min() - 1.5 * constantLow, 3)
        # print(str(i), maxValue, minValue, column.max(), column.min(), abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), str(printSet.index[0])[:-6], str(printSet.index[-1])[:-6])
        dataSet.plot(figsize=(10, 7), title='Anomaly Detection: Time period = ' + str(i + 1), xlabel='Time',
                     ylabel='Latency',
                     ylim=[minValue, maxValue],
                     xlim=[str(printSet.index[0])[:-6], str(printSet.index[-1])[:-6]])
        plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')
        # plt.plot(test_v1.index, test_v1[column'], color='green', label='Latency WO Smoothing')
        plt.plot(test.index, test['predictions'], color='orange', label='Predictions')
        plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '.' + ' Anomaly Detection- Time period ' + str(i + 1) + '.png', dpi=300,
                    bbox_inches='tight')
        plt.show()