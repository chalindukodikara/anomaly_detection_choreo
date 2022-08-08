import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from sklearn.covariance import oas
from statsmodels.tsa.arima.model import ARIMA
import arimaHelpers as arimaFunctions
import csv
import arimafd as oa
import warnings
warnings.filterwarnings("ignore") # Remove warning that appear when training
######################################################################
###################### ARIMAFD MODEL ###################################
######################################################################
def main(train, test, dataSetID, column, lowHighDifference, train_v1, test_v1):
    numOfSteps = 3 # Num of steps that need to forecasted
    # test = test[:8] # Test set size
    immediate1stPredictions = []
    immediate2ndPredictions = []
    immediate3rdPredictions = []
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)
    dataSetPred = dataSet.copy()

    print('\n')
    print('Low high difference : ', lowHighDifference)
    print('Training set size : ', len(train))
    print('Test set size : ', len(test))
    print('Dataset size : ', len(dataSet))
    print('\n')

    # Train the model
    # immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions. immediate2ndPredictions, immediate3rdPredictions)
    trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column, lowHighDifference, test_v1)
    #
    # # Add immediate predictions to the whole dataset
    # dataSetPred['predictions'] = np.nan
    # for i in range(len(immediate1stPredictions)):
    #     dataSetPred.loc[dataSetPred.head(len(dataSetPred)).index[i + len(train)], 'predictions'] = immediate1stPredictions[i]
    #
    # # Print dataframes
    # pd.set_option("display.max_rows", None, "display.max_columns", 9)
    # print('Print dataset')
    # print(dataSet)
    # print('Print testset with anomaly points marked')
    # print(testSetIsAnomaly)
    # fields = [str(dataSetID + 1) + '. Predictions and Anomalies']
    #
    # # Write anomaly table to csv file
    # with open(r'Anomaly.csv', 'a') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow(fields)
    #
    # testSetIsAnomaly.to_csv('Anomaly.csv', mode='a')
    #
    # with open(r'Anomaly.csv', 'a') as fd:
    #     writer = csv.writer(fd)
    #     writer.writerow('\n')
    #
    # # Create 3 dataframes for 3 observations
    # test1 = test.copy()
    # test2 = test.copy()
    # test3 = test.copy()
    #
    # # Remove 1st value from test1 set and remove first 2 values from test3, since immediate2ndPredictions doesnt contain 1st value of test set and immediate3rdPredictions doesnt contain 1st 2 values of test set
    # test2 = test2.drop(test2.index[0])
    # test3 = test3.drop(test3.index[[0, 1]])
    #
    # # Remove extra predictions
    # immediate2ndPredictions = immediate2ndPredictions[:-1]
    # immediate3rdPredictions = immediate3rdPredictions[:-2]
    #
    # # Add observation values into each test set
    # test1['predictions'] = immediate1stPredictions
    # test2['predictions'] = immediate2ndPredictions
    # test3['predictions'] = immediate3rdPredictions
    #
    # # Plots immediate predictions
    # # plotResults(dataSet, train, test1, models[0], 'ARIMA - Immediate 1st Predictions', 'ARIMA - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    # # plotResults(dataSet, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
    # #             'ARIMA - Immediate 2nd Predictions', 'ARIMA - Immediate 2nd Predictions in small time frame',
    # #             str(test1.index[0])[:-6], dataSetID)
    # # plotResults(dataSet, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
    # #             'ARIMA - Immediate 3rd Predictions', 'ARIMA - Immediate 3rd Predictions in small time frame',
    # #             str(test1.index[1])[:-6], dataSetID)
    #
    # # Plot final anomaly plots
    # plotAnomaly(testSetIsAnomaly, dataSet, train, dataSetID, test_v1, column)
    # # # plot the residuals of the model. The residuals are the difference between the original values and the predicted values from the model.
    # # result.resid.plot(kind='kde', title='residuals of the model')
    # # plt.show()
    #
    # print(str(dataSetID + 1) + '. Accuracy metrics')
    # with open('Accuracy.txt', 'a') as f:
    #     f.write(str(dataSetID + 1) + '. Accuracy metrics')
    #     f.write('\n')
    # # Get Accuracy values for fitted model and training set
    # print('Fitted model and training set')
    # with open('Accuracy.txt', 'a') as f:
    #     f.write('Fitted model and training set')
    #     f.write('\n')
    # arimaFunctions.forecastAccuracy(np.array(models[0].fittedvalues.values.tolist()), np.array(train[column]))
    #
    # # Get Accuracy values for immediate 1st values and test set
    # print('Immediate 1st values and test set')
    # with open('Accuracy.txt', 'a') as f:
    #     f.write('Immediate 1st values and test set')
    #     f.write('\n')
    # arimaFunctions.forecastAccuracy(np.array(test1['predictions']), np.array(test1[column]))
    #
    # # Get Accuracy values for immediate 2nd values and test set
    # print('Immediate 2nd values and test set')
    # with open('Accuracy.txt', 'a') as f:
    #     f.write('Immediate 2nd values and test set')
    #     f.write('\n')
    # arimaFunctions.forecastAccuracy(np.array(test2['predictions']), np.array(test2[column]))
    #
    # # Get Accuracy values for immediate 3rd values and test set
    # print('Immediate 3rd values and test set')
    # with open('Accuracy.txt', 'a') as f:
    #     f.write('Immediate 3rd values and test set')
    #     f.write('\n')
    # arimaFunctions.forecastAccuracy(np.array(test3['predictions']), np.array(test3[column]))


def trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column, lowHighDifference, test_v1):
    ###################### Initial Plotting ###############################
    ######################################################################
    # Find the suitable model using auto arima
    m = auto_arima(train[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
    print(m.summary())
    autoRegressive = m.get_params()['order'][0]
    numOfDifferences = m.get_params()['order'][1]
    movingAverage = m.get_params()['order'][2]

    # autoRegressive = 0
    # numOfDifferences = 1
    # movingAverage = 0

    # Plot train data and fitted model
    my_arima = oa.Arima_anomaly_detection(ar_order=autoRegressive)
    my_arima.fit(test[:20])
    ts_anomaly = my_arima.predict(test[20:])

    # my_array = np.random.normal(size=1000)  # init array
    # my_array[-3] = 1000  # init anomaly
    # ts = pd.DataFrame(my_array,
    #                   index=pd.date_range(start='01-01-2000',
    #                                       periods=1000,
    #                                       freq='H'))
    #
    # my_arima = oa.Arima_anomaly_detection(ar_order=3)
    # my_arima.fit(ts[:500])
    # ts_anomaly = my_arima.predict(ts[500:])
    pd.set_option("display.max_rows", None, "display.max_columns", 9)
    print(ts_anomaly)

    # train.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #            title='ARIMA - Train data and fitted model')
    # plt.plot(train.index, result.predict(1, len(train)), label='Fitted model')
    # plt.legend(loc='best')
    # plt.savefig(str(dataSetID) + '. ' + 'Train data and fitted model' + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # # Plot train data and fitted model in small time frame
    # # train.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    # #              title='ARIMA - Train data and fitted model in small time frame',
    # #              xlim=['2021-02-17 12:10', '2021-02-17 12:21'])
    # # plt.plot(train.index, result.predict(1, len(train)), label='Fitted model')
    # # plt.legend(loc='best')
    # # plt.show()
    # ######################## new variables ###############################
    # ######################################################################
    # models = []
    # train1 = train.copy() # Create new train dataframe to avoid changes in initial training dataset
    # train2 = train.copy() # New set to calculate mean
    # test1 = test.copy()
    # test1['predictions'] = np.nan
    # test1['is_anomaly'] = np.nan
    # test1['threshold'] = np.nan
    # test1['squared_error'] = np.nan
    # test1['threshold_upper'] = np.nan
    # test1['mean'] = np.nan
    # test1['threshold_lower'] = np.nan
    # # Predict immediate 3 values each time and add them into 3 arrays
    # for i in range(0, len(test)):
    #     ################### Find lowest and highest ##########################
    #     ######################################################################
    #     # # Define upper and lower threshold
    #     # # Find lowest and highest value in the normal dataset
    #     # lowest = 9999999
    #     # highest = -9999999
    #     # for j in range(len(train2)):
    #     #     if train2['in_avg_response_time'][j] < lowest:
    #     #         lowest = train2['in_avg_response_time'][j]
    #     #     # print('lowest: ', lowest)
    #     #     # print(dataSet_v1.iloc[i])
    #     #     elif train2['in_avg_response_time'][j] > highest:
    #     #         highest = train2['in_avg_response_time'][j]
    #     #     # print('highest: ', highest)
    #     #     # print(dataSet_v1.iloc[i])
    #     # lowHighDifference = round((highest - lowest), 5)
    #     # print('lowest and highest values ', lowest, highest, lowHighDifference)
    #
    #     ###################### Find parameters ###############################
    #     ######################################################################
    #     # Run auto arima after 20 data points
    #     if ((i+1)%20 == 0):
    #         m = auto_arima(train1[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
    #         print(m.summary())
    #         autoRegressive = m.get_params()['order'][0]
    #         numOfDifferences = m.get_params()['order'][1]
    #         movingAverage = m.get_params()['order'][2]
    #
    #     ###################### Train the model ###############################
    #     ######################################################################
    #     model = ARIMA(train1[column], order=(autoRegressive, numOfDifferences, movingAverage))
    #     result = model.fit()
    #     if i == 0 or i == 1 or i == 2:
    #         models.append(result)
    #     pred = result.get_forecast(steps=numOfSteps)
    #     immediate1stPredictions.append(pred.predicted_mean.values[0])
    #     immediate2ndPredictions.append(pred.predicted_mean.values[1])
    #     immediate3rdPredictions.append(pred.predicted_mean.values[2])
    #
    #     ###################### Define threshold ###############################
    #     ######################################################################
    #     meanValue = np.mean(train2[column].tolist())
    #
    #     # model_1 = ARIMA(train1[column], order=(0, 0, 0))
    #     # result_1 = model_1.fit()
    #     # pred_1 = result_1.get_forecast(steps=numOfSteps)
    #     if 0.5<meanValue and meanValue<1.5:
    #         multiplier_1 = 0.5
    #     elif  1.5<meanValue and meanValue<26:
    #         multiplier_1 = 0.5
    #     elif  45<meanValue and meanValue<70:
    #         multiplier_1 = 0.4
    #     elif  70<meanValue and meanValue<95:
    #         multiplier_1 = 0.15
    #     else:
    #         multiplier_1 = 0.25
    #
    #     thresholdUpper = round(meanValue + multiplier_1 * meanValue, 6)
    #     thresholdLower =  round(meanValue - multiplier_1 * meanValue , 6)
    #
    #
    #     squared_errors = (result.resid.values) ** 2
    #     threshold = np.mean(squared_errors) + 3 * np.std(squared_errors) + ((thresholdUpper - thresholdLower) * 0.75) ** 2
    #     # + ((thresholdUpper - thresholdLower) / 2) ** 2
    #     # lowHighDifference
    #     print('mean, std',str(test.index[i])[:-6], threshold, np.mean(squared_errors), 3 * np.std(squared_errors), (thresholdUpper - thresholdLower) ** 2)
    #     ######################### Find anomaly ###############################
    #     ######################################################################
    #     # Integer value is returned, not string
    #     anomaly, condition, squaredError = detectAnomaly(pred.predicted_mean.values[0], i, test, threshold, test_v1, thresholdUpper, thresholdLower)
    #
    #     # train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
    #
    #     # If there is an anomaly add the prediction to the training set
    #     # if anomaly == 1 and condition == 'above_upper_threshold':
    #     #     x = test.copy()
    #     #     x = x[i:i + 1]
    #     #     x.iloc[0, 0] = pred.predicted_mean.values[0]
    #     #     train1 = pd.concat([train1, x], ignore_index=False, axis=0)
    #     #     immediate1stPredictions.append(pred.predicted_mean.values[0])
    #     #     immediate2ndPredictions.append(pred.predicted_mean.values[1])
    #     #     immediate3rdPredictions.append(pred.predicted_mean.values[2])
    #     #     test1.iloc[i, 1] = pred.predicted_mean.values[0]
    #     #     print('threshold 1', str(test.index[i])[:-6], threshold, np.mean(squared_errors), 1 * np.std(squared_errors),
    #     #           lowHighDifference, pred.predicted_mean.values[0])
    #     #
    #     # elif anomaly == 1 and condition == 'below_lower_threshold':
    #     #     x = test.copy()
    #     #     x = x[i:i + 1]
    #     #     x.iloc[0, 0] = pred.predicted_mean.values[0]
    #     #     train1 = pd.concat([train1, x], ignore_index=False, axis=0)
    #     #     immediate1stPredictions.append(pred.predicted_mean.values[0])
    #     #     immediate2ndPredictions.append(pred.predicted_mean.values[1])
    #     #     immediate3rdPredictions.append(pred.predicted_mean.values[2])
    #     #     test1.iloc[i, 1] = pred.predicted_mean.values[0]
    #     #     print('threshold 2', str(test.index[i])[:-6], threshold, np.mean(squared_errors), 1 * np.std(squared_errors),
    #     #           lowHighDifference, pred.predicted_mean.values[0])
    #     #
    #     # else:
    #     #     train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
    #     #     immediate1stPredictions.append(pred.predicted_mean.values[0])
    #     #     immediate2ndPredictions.append(pred.predicted_mean.values[1])
    #     #     immediate3rdPredictions.append(pred.predicted_mean.values[2])
    #     #     test1.iloc[i, 1] = pred.predicted_mean.values[0]
    #     #     print('threshold 3', str(test.index[i])[:-6], threshold, np.mean(squared_errors), 1 * np.std(squared_errors),
    #     #           lowHighDifference, pred.predicted_mean.values[0])
    #
    #     if anomaly == 1:
    #         x = test.copy()
    #         x = x[i:i + 1]
    #         x.iloc[0, 0] = pred.predicted_mean.values[0]
    #         # train1 = pd.concat([train1, x], ignore_index=False, axis=0) # anomaly point is not added to training set
    #         train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0) # anomaly point is added to training set
    #
    #         # Train2 set to calculate mean, train 2 can contain 100 data points
    #         x.iloc[0, 0] = meanValue
    #         train2 = pd.concat([train2, x], ignore_index=False, axis=0)
    #         if len(train2) != 100:
    #             pass
    #         else:
    #             train2 = train2[1:]
    #     else:
    #         train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
    #         train2 = pd.concat([train2, test[i:i + 1]], ignore_index=False, axis=0)
    #         if len(train2) != 100:
    #            pass
    #         else:
    #             train2 = train2[1:]
    #
    #     # Check train 1 size and try to hold the size at 60
    #     if len(train1) != 60:
    #         pass
    #     else:
    #         train1 = train1[1:]
    #
    #     test1.iloc[i, 1] = pred.predicted_mean.values[0]
    #     test1.iloc[i, 2] = str(anomaly)
    #     test1.iloc[i, 3] = threshold
    #     test1.iloc[i, 4] = squaredError
    #     test1.iloc[i, 5] = thresholdUpper
    #     test1.iloc[i, 6] = meanValue
    #     test1.iloc[i, 7] = thresholdLower
    # return test1, models

def detectAnomaly(predictedValue, i, test, threshold, test_v1, thresholdUpper, thresholdLower):
    squaredError = (predictedValue - test.iloc[i, 0]) ** 2
    squaredError_1 = (predictedValue - test_v1.iloc[i, 0]) ** 2
    # print('Squared error, threshold', squaredError,squaredError_1, threshold)
    condition = ''
    if predictedValue > thresholdUpper and test.iloc[i, 0] > thresholdUpper:
        anomaly = 1
        condition = 'above_upper_threshold'
    elif predictedValue < thresholdLower and test.iloc[i, 0] < thresholdLower:
        anomaly = 1
        condition = 'below_lower_threshold'
    else:
        anomaly = (squaredError >= threshold).astype(int)
        condition = 'normal_threshold'
    # anomaly = (squaredError >= threshold).astype(int)
    return anomaly, condition, squaredError

def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations


    # index = -13
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency', title=title1, xlim=[str(trainSet.index[-3])[:-9], str(testSet.index[int((len(testSet)/6)*5)])[:-9]])
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
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
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
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
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()

def plotAnomaly(test, dataSet, train, dataSetID, test_v1, column):
    dataSetWithAnomalies = pd.concat([train, test], ignore_index=False, axis=0)
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

    printSet_2 = dataSetWithAnomalies.copy()
    printSet_2 = printSet_2.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) - 5))])[:-9]]
    columnUpperThreshold = printSet_2['threshold_upper']
    columnLowerThreshold = printSet_2['threshold_lower']
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
        if columnValues.max() < columnUpperThreshold.max():
            maxValue = round(columnUpperThreshold.max() + 0.5 * constantHigh, 3)
        else:
            maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)

        if columnValues.min() > columnLowerThreshold.min():
            minValue = round(columnLowerThreshold.min() - 0.5 * constantLow, 3)
        else:
            minValue = round(columnValues.min() - 0.5 * constantLow, 3)
    else:
        if columnValues.max() < columnUpperThreshold.max():
            maxValue = round(columnUpperThreshold.max() + 1.5 * constantHigh, 3)
        else:
            maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)

        if columnValues.min() > columnLowerThreshold.min():
            minValue = round(columnLowerThreshold.min() - 1.5 * constantLow, 3)
        else:
            minValue = round(columnValues.min() - 1.5 * constantLow, 3)
        # maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
        # minValue = round(columnValues.min() - 1.5 * constantLow, 3)

    # print(maxValue, minValue, column.max(), column.min())
    # dataSet.plot(figsize=(10, 7), title='Anomaly whole test set 1', xlabel='Time', ylabel='Latency',
    #              ylim=[minValue, maxValue],
    #              xlim=[str(train.index[-13])[:-9], str(test.index[int((len(test) - 1))])[:-9]])
    dataSet.plot(figsize=(10, 7), title='Anomaly whole test set 1', xlabel='Time', ylabel='Latency',
                 ylim=[minValue, maxValue],
                 xlim=[str(train.index[-3])[:-9], str(test.index[int((len(test) - 1))])[:-9]])
    plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')

    # plt.plot(test_v1.index, test_v1['in_avg_response_time'], color='green', label='Latency WO Smoothing')
    plt.plot(test.index, test['threshold_upper'], color='k', label='threshold_upper', linestyle='--')
    plt.plot(test.index, test['threshold_lower'], color='k', label='threshold_upper', linestyle='--')
    plt.plot(test.index, test['mean'], color='green', label='mean', linestyle='--')
    plt.plot(test.index, test['predictions'], color='orange', label='Predictions')

    plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '.' + ' Anomaly whole test set 1.png', dpi=300, bbox_inches='tight')
    plt.show()

    printSet = dataSet.copy()
    # printSet = printSet.loc[str(train.index[-13])[:-9]:str(test.index[int((len(test) / 3) * 1)])[:-9]]
    printSet = printSet.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) / 3) * 1)])[:-9]]
    columnValues = printSet[column]
    printSet_2 = dataSetWithAnomalies.copy()
    printSet_2 = printSet_2.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) - 5))])[:-9]]
    columnUpperThreshold = printSet_2['threshold_upper']
    columnLowerThreshold = printSet_2['threshold_lower']

    if columnValues.max() > 1:
        constantHigh = 1.5
    else:
        constantHigh = 0.5
    if abs(columnValues.min()) > 1:
        constantLow = 2
    else:
        constantLow = 0.3
    if columnValues.max() < 2:
        if columnValues.max() < columnUpperThreshold.max():
            maxValue = round(columnUpperThreshold.max() + 0.5 * constantHigh, 3)
        else:
            maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)

        if columnValues.min() > columnLowerThreshold.min():
            minValue = round(columnLowerThreshold.min() - 0.5 * constantLow, 3)
        else:
            minValue = round(columnValues.min() - 0.5 * constantLow, 3)
    else:
        if columnValues.max() < columnUpperThreshold.max():
            maxValue = round(columnUpperThreshold.max() + 1.5 * constantHigh, 3)
        else:
            maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)

        if columnValues.min() > columnLowerThreshold.min():
            minValue = round(columnLowerThreshold.min() - 1.5 * constantLow, 3)
        else:
            minValue = round(columnValues.min() - 1.5 * constantLow, 3)
        # maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
        # minValue = round(columnValues.min() - 1.5 * constantLow, 3)

    # print(maxValue, minValue, column.max(), column.min())
    # dataSet.plot(figsize=(10, 7), title='Predictions for test set values 2', xlabel='Time', ylabel='Latency',
    #              ylim=[minValue, maxValue],
    #              xlim=[str(train.index[-13])[:-9], str(test.index[int((len(test) / 3) * 1)])[:-9]])
    dataSet.plot(figsize=(10, 7), title='Predictions for test set values 2', xlabel='Time', ylabel='Latency',
                 ylim=[minValue, maxValue],
                 xlim=[str(train.index[-3])[:-9], str(test.index[int((len(test) / 3) * 1)])[:-9]])
    plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')
    # plt.plot(test_v1.index, test_v1['in_avg_response_time'], color='green', label='Latency WO Smoothing')
    plt.plot(test.index, test['threshold_upper'], color='k', label='threshold_upper', linestyle='--')
    plt.plot(test.index, test['threshold_lower'], color='k', label='threshold_upper', linestyle='--')
    plt.plot(test.index, test['mean'], color='green', label='mean', linestyle='--')
    plt.plot(test.index, test['predictions'], color='orange', label='Predictions')
    plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '.' + ' Anomaly whole test set 2.png', dpi=300, bbox_inches='tight')
    plt.show()

    for i in range(int(len(test) / 30)):
        printSet = dataSet.copy()
        printSet = printSet.loc[str(test.index[i * 30])[:-9]:str(test.index[i * 30 + 30])[:-9]]
        columnValues = printSet[column]
        printSet_2 = dataSetWithAnomalies.copy()
        printSet_2 = printSet_2.loc[str(train.index[-3])[:-9]:str(test.index[int((len(test) - 5))])[:-9]]
        columnUpperThreshold = printSet_2['threshold_upper']
        columnLowerThreshold = printSet_2['threshold_lower']
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
            if columnValues.max() < columnUpperThreshold.max():
                maxValue = round(columnUpperThreshold.max() + 0.5 * constantHigh, 3)
            else:
                maxValue = round(columnValues.max() + 0.5 * constantHigh, 3)

            if columnValues.min() > columnLowerThreshold.min():
                minValue = round(columnLowerThreshold.min() - 0.5 * constantLow, 3)
            else:
                minValue = round(columnValues.min() - 0.5 * constantLow, 3)
        else:
            if columnValues.max() < columnUpperThreshold.max():
                maxValue = round(columnUpperThreshold.max() + 1.5 * constantHigh, 3)
            else:
                maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)

            if columnValues.min() > columnLowerThreshold.min():
                minValue = round(columnLowerThreshold.min() - 1.5 * constantLow, 3)
            else:
                minValue = round(columnValues.min() - 1.5 * constantLow, 3)
            # maxValue = round(columnValues.max() + 1.5 * constantHigh, 3)
            # minValue = round(columnValues.min() - 1.5 * constantLow, 3)
        # print(str(i), maxValue, minValue, column.max(), column.min(), abs(dataSet.iloc[0, 0] - dataSet.iloc[1, 0]), str(printSet.index[0])[:-6], str(printSet.index[-1])[:-6])
        dataSet.plot(figsize=(10, 7), title='Anomaly Detection: Time period = ' + str(i + 1), xlabel='Time',
                     ylabel='Latency',
                     ylim=[minValue, maxValue],
                     xlim=[str(printSet.index[0])[:-6], str(printSet.index[-1])[:-6]])
        plt.axvline(pd.to_datetime(str(train.index[-1])[:-6]), color='k', linestyle='--')
        # plt.plot(test_v1.index, test_v1[column'], color='green', label='Latency WO Smoothing')
        plt.plot(test.index, test['threshold_upper'], color='k', label='threshold_upper', linestyle='--')
        plt.plot(test.index, test['threshold_lower'], color='k', label='threshold_upper', linestyle='--')
        plt.plot(test.index, test['mean'], color='green', label='mean', linestyle='--')
        plt.plot(test.index, test['predictions'], color='orange', label='Predictions')
        plt.scatter(testAnomaly.index, testAnomaly[column], marker='o', color='red', label='Anomaly')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '.' + ' Anomaly Detection- Time period ' + str(i + 1) + '.png', dpi=300,
                    bbox_inches='tight')
        plt.show()