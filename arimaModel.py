import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import arimaHelpers as arimaFunctions


######################################################################
###################### ARIMA MODEL ###################################
######################################################################

def main(train, test, dataSetID, column):

    numOfDifferences = 1
    numOfSteps = 3
    # test = test[:50]
    immediate1stPredictions = []
    immediate2ndPredictions = []
    immediate3rdPredictions = []
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)

    # View the trend and seasonality
    # arimaFunctions.decomposeData(train, train[column])

    # Check stationary
    # arimaFunctions.checkStationary(train[column])

    # Dickey Fuller test to find how many differences required
    # arimaFunctions.dickeyFullerTest(train)

    # Take difference to check how many number of differences required
    # arimaFunctions.takeDifference(train, numOfDifferences)

    # Find the required model
    # m = auto_arima(train[column], seasonal=False, m=0,max_p=7, max_d=5,max_q=7, max_P=4, max_D=4,max_Q=4)
    # print(type(m))
    # print(m.get_params()['order'])
    # print(m.get_params()['order'][0])

    # Train the model
    # immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions. immediate2ndPredictions, immediate3rdPredictions)
    models = trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column)

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

    # print(len(immediate2ndPredictions))
    # print(len(immediate3rdPredictions))
    # print(str(train.index[-1])[:-6])
    # print(str(test1.index[0])[:-6])
    # print(str(test1.index[1])[:-6])

    # pd.set_option("display.max_rows", None, "display.max_columns", None)
    #
    # print(test1)
    # print(test2)
    # print(test3)

    plotResults(dataSet, train, test1, models[0], 'ARIMA - Immediate 1st Predictions', 'ARIMA - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
                'ARIMA - Immediate 2nd Predictions', 'ARIMA - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
                'ARIMA - Immediate 3rd Predictions', 'ARIMA - Immediate 3rd Predictions in small time frame',
                str(test1.index[1])[:-6], dataSetID)

    # # plot the residuals of the model. The residuals are the difference between the original values and the predicted values from the model.
    # result.resid.plot(kind='kde', title='residuals of the model')
    # plt.show()

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

    # prediction_train = result.predict(start=0,end=397)
    #
    # # Get predictions for test set
    # testSetSize = 30
    # pred = result.get_forecast(steps=testSetSize)
    # test['predicted'] = pred.predicted_mean.values
    # test['lower_latency'] = pred.conf_int(alpha=0.05).values[:, 0:1]
    # test['higher_latency'] = pred.conf_int(alpha=0.05).values[:, 1:2]
    #
    # # Plot final results for ARIMA
    # arimaFunctions.plotFinalResult(train, test, result)
    #
    # # Plot test and predictions
    # arimaFunctions.plotAnomaly(train, test, result, pred)

def trainModel(train, test, numOfSteps, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions, dataSetID, column):
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
    train.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title='ARIMA - Train data and fitted model in small time frame',
                 xlim=['2021-02-17 12:10', '2021-02-17 12:21'])
    plt.plot(train.index, result.predict(1, len(train)), label='Fitted model')
    plt.legend(loc='best')
    plt.show()

    models = []
    train1 = train.copy() # Create new train dataframe to avoid changes in initial training dataset

    # print(test)

    # Predict immediate 3 values each time and add them into 3 arrays
    for i in range(len(test)):
        # Create the model and fit it
        # print(i)
        # print(test[i:i + 1])
        if ((i+1)%10 == 0):
            m = auto_arima(train1[column], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
            print(m.summary())
            autoRegressive = m.get_params()['order'][0]
            numOfDifferences = m.get_params()['order'][1]
            movingAverage = m.get_params()['order'][2]
        model = ARIMA(train1[column], order=(autoRegressive, numOfDifferences, movingAverage))
        result = model.fit()
        if i == 0 or i == 1 or i == 2:
            models.append(result)
        # Print summary of the model
        # print(result.summary())
        pred = result.get_forecast(steps=numOfSteps)
        immediate1stPredictions.append(pred.predicted_mean.values[0])
        immediate2ndPredictions.append(pred.predicted_mean.values[1])
        immediate3rdPredictions.append(pred.predicted_mean.values[2])
        # train.concat(test.iloc[0])
        train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
        # train1[column] = savgol_filter(train1[column], 15, 8)
    return models

def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations
    startDate = '2021-02-17 12:18'
    endDate = '2021-02-17 12:40'

    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency', title=title1, xlim=[str(trainSet.index[-13])[:-9], str(testSet.index[int((len(testSet)/6)*5)])[:-9]])
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title1 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot whole dataset, fitted values and immediate observations in small time frame
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-11])[:-9], str(testSet.index[int(len(testSet)/3)])[:-9]])
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
                 xlim=[str(trainSet.index[-9])[:-9], str(testSet.index[int(len(testSet)/6)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()

    ###################################################################################################
    ###################################################################################################
    # # Plot whole dataset, fitted values and immediate observations in small time frame
    # dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #              title=title2 + 'sadsa',
    #              xlim=[str(testSet.index[-180])[:-9], str(testSet.index[-130])[:-9]])
    # plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    # plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    # plt.legend(loc='best')
    # plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # plt.figure(figsize=(10, 7))
    # plt.xlabel('Time')
    # plt.ylabel('Latency')
    # plt.title(title1)
    # start = np.datetime64(str(testSet.index[-180])[:-9], 'ns')
    # end = np.datetime64(str(testSet.index[-130])[:-9], 'ns')
    # plt.xlim([start, end])
    # plt.scatter(dataSet.index, dataSet['in_avg_response_time'], label = 'original data')
    # plt.scatter(testSet.index, testSet['predictions'], label='predictions')
    # plt.legend(loc='best')
    # plt.show()

