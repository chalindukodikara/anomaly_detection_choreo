import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from ThymeBoost import ThymeBoost as tb
import arimaHelpers as arimaFunctions

def main(train, test, dataSetID, column):
    immediate1stPredictions = []
    immediate2ndPredictions = []
    immediate3rdPredictions = []
    # test = test[:12]
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)


    train =  train[column].squeeze()
    test = test[column].squeeze()
    trainModel(train, test, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions,dataSetID)

    train = train.to_frame()
    test = test.to_frame()

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
    models = [0,1,2]


    plotResults(dataSet, train, test1, models[0], 'ThymeBoost - Immediate 1st Predictions',
                'ThymeBoost - Immediate 1st Predictions in small time frame', str(train.index[-1])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:1]], ignore_index=False, axis=0), test2, models[1],
                'ThymeBoost - Immediate 2nd Predictions', 'ThymeBoost - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID)
    plotResults(dataSet, pd.concat([train, test[0:2]], ignore_index=False, axis=0), test3, models[2],
                'ThymeBoost - Immediate 3rd Predictions', 'ThymeBoost - Immediate 3rd Predictions in small time frame',
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


def trainModel(train, test, immediate1stPredictions, immediate2ndPredictions, immediate3rdPredictions,dataSetID):
    boosted_model = tb.ThymeBoost(verbose=0)
    output = boosted_model.autofit(train, seasonal_period=0)
    # output = boosted_model.fit(train,
    #                   trend_estimator='linear',
    #                   seasonal_estimator='fourier',
    #                   seasonal_period=25,
    #                   split_cost='mse',
    #                   global_cost='maicc',
    #                   fit_type='global')
    boosted_model.plot_results(output)
    boosted_model.plot_components(output)
    predictions = boosted_model.predict(output, 3)

    models = []
    train1 = train.copy()  # Create new train dataframe to avoid changes in initial training dataset

    # Predict immediate 3 values each time and add them into 3 arrays
    for i in range(len(test)):
        # Create the model and fit it
        # print(i)
        # print(test[i:i + 1])
        boosted_model = tb.ThymeBoost(verbose=0)
        result = boosted_model.autofit(train1, seasonal_period=0)
        if i == 0 or i == 1 or i == 2:
            models.append(result)
        # Print summary of the model
        # print(result.summary())
        predictions = boosted_model.predict(result, 3)
        immediate1stPredictions.append(round(predictions['predictions'][0], 5))
        immediate2ndPredictions.append(round(predictions['predictions'][1], 5))
        immediate3rdPredictions.append(round(predictions['predictions'][2], 5))
        # train.concat(test.iloc[0])
        train1 = pd.concat([train1, test[i:i + 1]], ignore_index=False, axis=0)
        # train1[column] = savgol_filter(train1[column], 15, 8)

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
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
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
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
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
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    plt.legend(loc='best')
    plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    plt.show()