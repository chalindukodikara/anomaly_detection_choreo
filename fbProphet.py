import fbprophet
import pandas as pd
import matplotlib.pyplot as plt
import arimaHelpers as arimaFunctions
import numpy as np


def main(dataSet_v2, dataSetID, size, column):
    immediate1stPredictions = [[], [], []]
    immediate2ndPredictions = [[], [], []]
    immediate3rdPredictions = [[], [], []]

    # Set columns as ds and y
    dataSet_v3 = dataSet_v2.copy()
    dataSet_v3 = dataSet_v3.reset_index()
    # taxi_df = dataSet_v3.reset_index()[['timestamp', 'value']].rename({'timestamp': 'ds', 'value': 'y'}, axis='columns')
    dataSet_v3.columns = ['ds', 'y']
    print(dataSet_v3)

    # print version number
    print('Prophet %s' % fbprophet.__version__)

    # Convert string column to datetime
    dataSet_v3['ds'] = pd.to_datetime(dataSet_v3['ds'])

    print(dataSet_v3.dtypes)

    ######### Converting to daily time series##########
    dataSet_v5 = dataSet_v2.copy()
    dataSet_v5 = dataSet_v5[0:1]
    for i in range(1, len(dataSet_v3)):
        idx = dataSet_v5.tail(1).index[0] + pd.Timedelta(weeks=4)
        dataSet_v5.loc[idx] = round(dataSet_v2.iloc[i, 0], 6)
    dataSet_v5 = dataSet_v5.reset_index()
    # taxi_df = dataSet_v3.reset_index()[['timestamp', 'value']].rename({'timestamp': 'ds', 'value': 'y'}, axis='columns')
    dataSet_v5.columns = ['ds', 'y']
    dataSet_v5['ds'] = pd.to_datetime(dataSet_v5['ds'])
    print(dataSet_v5)

    # REMOVING THE TIMEZONE INFORMATION
    dataSet_v4 = dataSet_v3.copy()
    dataSet_v4['ds'] = dataSet_v3['ds'].dt.tz_localize(None)
    dataSet_v5['ds'] = dataSet_v5['ds'].dt.tz_localize(None)
    # PRINT THE DATATYPE OF EACH COLUMN BEFORE
    # MANIPULATION
    # print(dataSet_v3)



    # train, test split
    train, test = dataSet_v4[0:size], dataSet_v4[size:len(dataSet_v4)]
    train_1, test_1 = dataSet_v5[0:size], dataSet_v5[size:len(dataSet_v4)]


    # test = test[:20]
    # Train model
    models = trainModel(dataSet_v3, train_1, test_1, size, dataSetID, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions)

    print(immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions)

    # Create 3 dataframes for 3 observations
    test1 = test.copy()
    test2 = test.copy()
    test3 = test.copy()

    # Remove 1st value from test1 set and remove first 2 values from test3, since immediate2ndPredictions doesnt contain 1st value of test set and immediate3rdPredictions doesnt contain 1st 2 values of test set
    test2 = test2.drop(test2.index[0])
    test3 = test3.drop(test3.index[[0, 1]])

    # Remove extra predictions
    immediate2ndPredictions[0] = immediate2ndPredictions[0][:-1]
    immediate2ndPredictions[1] = immediate2ndPredictions[1][:-1]
    immediate2ndPredictions[2] = immediate2ndPredictions[2][:-1]

    immediate3rdPredictions[0] = immediate3rdPredictions[0][:-2]
    immediate3rdPredictions[1] = immediate3rdPredictions[1][:-2]
    immediate3rdPredictions[2] = immediate3rdPredictions[2][:-2]

    # Add observation values into each test set
    test1['predictions_lower'] =  immediate1stPredictions[0]
    test1['predictions'] = immediate1stPredictions[1]
    test1['predictions_upper'] = immediate1stPredictions[2]

    test2['predictions_lower'] = immediate2ndPredictions[0]
    test2['predictions'] = immediate2ndPredictions[1]
    test2['predictions_upper'] = immediate2ndPredictions[2]

    test3['predictions_lower'] = immediate3rdPredictions[0]
    test3['predictions'] = immediate3rdPredictions[1]
    test3['predictions_upper'] = immediate3rdPredictions[2]

    test1['ds'] = test1['ds'].dt.tz_localize('UTC')
    test2['ds'] = test2['ds'].dt.tz_localize('UTC')
    test3['ds'] = test3['ds'].dt.tz_localize('UTC')
    test1.set_index('ds', inplace=True)
    test2.set_index('ds', inplace=True)
    test3.set_index('ds', inplace=True)

    print(test1[['y', 'predictions']].head(20))
    print(test2[['y', 'predictions']])
    print(test3[['y', 'predictions']])

    train['ds'] = train['ds'].dt.tz_localize('UTC')
    test['ds'] = test['ds'].dt.tz_localize('UTC')
    train.set_index('ds', inplace=True)
    test.set_index('ds', inplace=True)

    plotResults(dataSet_v2, train, test1, models[0], 'FB Prophet - Immediate 1st Predictions',
                'FB Prophet - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    plotResults(dataSet_v2, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
                'FB Prophet - Immediate 2nd Predictions', 'FB Prophet - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID)
    plotResults(dataSet_v2, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
                'FB Prophet - Immediate 3rd Predictions', 'FB Prophet - Immediate 3rd Predictions in small time frame',
                str(test1.index[1])[:-6], dataSetID)

    print(str(dataSetID + 1) + '. Accuracy metrics')
    with open('Accuracy.txt', 'a') as f:
        f.write(str(dataSetID + 1) + '. Accuracy metrics')
        f.write('\n')
    # Get Accuracy values for fitted model and training set
    print('Fitted model and training set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Fitted model and training set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(models[0]['yhat']),
                                    np.array(train['y']))

    # Get Accuracy values for immediate 1st values and test set
    print('Immediate 1st values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 1st values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test1['predictions']), np.array(test1['y']))

    # Get Accuracy values for immediate 2nd values and test set
    print('Immediate 2nd values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 2nd values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test2['predictions']), np.array(test2['y']))

    # Get Accuracy values for immediate 3rd values and test set
    print('Immediate 3rd values and test set')
    with open('Accuracy.txt', 'a') as f:
        f.write('Immediate 3rd values and test set')
        f.write('\n')
    arimaFunctions.forecastAccuracy(np.array(test3['predictions']), np.array(test3['y']))


def trainModel(dataSet_v3, train, test, size, dataSetID, immediate1stPredictions, immediate2ndPredictions,
               immediate3rdPredictions):
    # define the model, confidence interval = 95%. Default 80%. changepoint_range
    model = fbprophet.Prophet(interval_width=0.95)
    model.fit(train)

    future = model.make_future_dataframe(periods=0, include_history=True, freq='1 m')
    forecast = model.predict(future)
    print(forecast)
    model.plot_components(forecast)
    forecast_2 = forecast.copy()
    forecast_2 = forecast_2.drop(
        ['trend', 'trend_lower', 'trend_upper', 'additive_terms', 'additive_terms_lower', 'additive_terms_upper',
         'multiplicative_terms', 'multiplicative_terms_lower', 'multiplicative_terms_upper'], axis=1)
    # fig1 = model.plot(forecast)
    # plt.plot(forecast['ds'], forecast['yhat'])
    # plt.show()

    result1 = pd.concat([train.set_index('ds')['y'], forecast.set_index('ds')[['yhat', 'yhat_lower', 'yhat_upper']]],
                        axis=1)

    # Print train and fitted model
    dataSetPlot = dataSet_v3.set_index('ds', inplace=False)
    dataSetPlot[0:size].plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                             title='FB Prophet - Train data and fitted model')
    plt.plot(dataSetPlot[0:size].index, result1['yhat'], label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + 'Train data and fitted model' + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    models = []
    models.append(forecast)
    train1 = train.copy()  # Create new train dataframe to avoid changes in initial training dataset

    # print(test)

    # Predict immediate 3 values each time and add them into 3 arrays
    for i in range(len(test)):
        # Create the model and fit it
        model = fbprophet.Prophet(interval_width=0.95)
        result = model.fit(train1)
        if i == 0 or i == 1 or i == 2:
            models.append(result)

        future = model.make_future_dataframe(periods=3, include_history=False, freq='1 m')
        forecast = model.predict(future)
        # multi dimensional arrays for upper and lower values
        immediate1stPredictions[0].append(round(forecast.iloc[0]['yhat_lower'], 5))
        immediate1stPredictions[1].append(round(forecast.iloc[0]['yhat'], 5))
        immediate1stPredictions[2].append(round(forecast.iloc[0]['yhat_upper'], 5))

        immediate2ndPredictions[0].append(round(forecast.iloc[1]['yhat_lower'], 5))
        immediate2ndPredictions[1].append(round(forecast.iloc[1]['yhat'], 5))
        immediate2ndPredictions[2].append(round(forecast.iloc[1]['yhat_upper'], 5))

        immediate3rdPredictions[0].append(round(forecast.iloc[2]['yhat_lower'], 5))
        immediate3rdPredictions[1].append(round(forecast.iloc[2]['yhat'], 5))
        immediate3rdPredictions[2].append(round(forecast.iloc[2]['yhat_upper'], 5))
        # train.concat(test.iloc[0])
        train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
    # train1['in_avg_response_time'] = savgol_filter(train1['in_avg_response_time'], 15, 8)
    return models


def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations
    startDate = '2021-02-17 12:18'
    endDate = '2021-02-17 12:40'

    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency', title=title1,
                 xlim=[str(trainSet.index[-8])[:-9], str(testSet.index[int((len(testSet) / 3) * 2)])[:-9]])
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title1 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Plot whole dataset, fitted values and immediate observations in small time frame
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-6])[:-9], str(testSet.index[int(len(testSet) / 3)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + '.png', dpi=300, bbox_inches='tight')
    plt.show()

    #
    # Plot whole dataset, fitted values and immediate observations in small time frame
    dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                 title=title2,
                 xlim=[str(trainSet.index[-5])[:-9], str(testSet.index[int(len(testSet) / 6)])[:-9]])
    plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()
