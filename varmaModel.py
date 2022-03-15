import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pmdarima import auto_arima
from statsmodels.tsa.arima.model import ARIMA
import arimaHelpers as arimaFunctions
import csv
import arimaHelpers
import pandas as pd
from statsmodels.tsa.statespace.varmax import VARMAX
from statsmodels.tsa.stattools import adfuller
from sklearn import metrics
from timeit import default_timer as timer
import time
import warnings
warnings.filterwarnings("ignore")
######################################################################
###################### VARMA MODEL ###################################
######################################################################

def main(train, test, dataSetID, columnList, train_v1, test_v1):
    numOfSteps = 3
    # test = test[:20]
    immediate1stPredictionsLatency = []
    immediate1stPredictionsThroughput = []
    immediate1stPredictionsCPU = []
    immediate1stPredictionsMemory = []
    numOfDifferences = 0
    dataSet = pd.concat([train, test], ignore_index=False, axis=0)
    print('Training set size : ', len(train))
    print('Test set size : ', len(test))
    print('Dataset size : ', len(dataSet))

    # train_diff, numOfDifferences = makeStationary(train, columnList)
    # print("After calling makeStationary, number of differences :- ", numOfDifferences)
    # print(train_diff.info())
    #
    # # print(invert_difference(train, train_diff, numOfDifferences))
    # # print(train_org)
    # # print(train)
    # p, q = findBestParameters(train, train_diff, test[:30], columnList)
    # print(p, q)

    models, test_diff, dataSet_diff, train_diff = trainModel(train, test, numOfSteps, immediate1stPredictionsLatency, immediate1stPredictionsThroughput, immediate1stPredictionsCPU, immediate1stPredictionsMemory,
               dataSetID, columnList, test_v1)
    print('called train model')
    print(immediate1stPredictionsLatency)
    # Create 3 dataframes for 3 observations
    test1 = test_diff.copy()
    test2 = test_diff.copy()
    test3 = test_diff.copy()

    # Remove 1st value from test1 set and remove first 2 values from test3, since immediate2ndPredictions doesnt contain 1st value of test set and immediate3rdPredictions doesnt contain 1st 2 values of test set
    test2 = test2.drop(test2.index[0])
    test3 = test3.drop(test3.index[[0, 1]])

    # Remove extra predictions
    # immediate2ndPredictions = immediate2ndPredictions[:-1]
    # immediate3rdPredictions = immediate3rdPredictions[:-2]

    # Add observation values into each test set
    test1['predictionsLatency'] = immediate1stPredictionsLatency
    test1['predictionsThroughput'] = immediate1stPredictionsThroughput
    test1['predictionsCPU'] = immediate1stPredictionsCPU
    test1['predictionsMemory'] = immediate1stPredictionsMemory
    # test2['predictions'] = immediate2ndPredictions
    # test3['predictions'] = immediate3rdPredictions

    plotResults(dataSet_diff, train_diff, test1, models[0], 'ARIMA - Immediate 1st Predictions',
                'ARIMA - Immediate 1st Predictions in small time frame', str(train_diff.index[-1])[:-6], dataSetID, columnList)
    plotResults(dataSet_diff, pd.concat([train_diff, test_diff[0:1]], ignore_index=False, axis=0), test2, models[1],
                'ARIMA - Immediate 2nd Predictions', 'ARIMA - Immediate 2nd Predictions in small time frame',
                str(test1.index[0])[:-6], dataSetID, columnList)
    plotResults(dataSet_diff, pd.concat([train_diff, test_diff[0:2]], ignore_index=False, axis=0), test3, models[2],
                'ARIMA - Immediate 3rd Predictions', 'ARIMA - Immediate 3rd Predictions in small time frame',
                str(test1.index[1])[:-6], dataSetID, columnList)

def makeStationary(dataSet, columnList):
    # # Make the whole series stationary, all the columns will be stationary
    # dataSet_diff = dataSet.copy()
    # print("Before taking the differences")
    # print(dataSet_diff.info())
    # stationary = [False, False, False, False]
    # numOfDifferences = 0
    # # True = 1, False = 0. So sum gives a value.
    # while sum(stationary) != 4:
    #     x = 0
    #     for name, column in dataSet_diff[columnList].iteritems():
    #         is_stationary = arimaHelpers.stationaryCheck(dataSet_diff[name], name)
    #         stationary[x] = is_stationary
    #         print(name, stationary)
    #         print('\n')
    #         x += 1
    #
    #     if sum(stationary) != 4:
    #         numOfDifferences += 1
    #         dataSet_diff = dataSet.diff(periods=numOfDifferences)
    #         dataSet_diff.dropna(inplace=True)  # Drop null values
    #
    #
    #
    # print("After taking the differences", numOfDifferences)
    # print(dataSet_diff.info())
    # print(dataSet_diff, len(dataSet_diff))
    # return dataSet_diff, numOfDifferences

    diffList = []
    for i in range(len(columnList)):
        m = auto_arima(dataSet[columnList[i]], seasonal=False, m=0, max_p=7, max_d=5, max_q=7, max_P=4, max_D=4, max_Q=4)
        diff = m.get_params()['order'][1]
        diffList.append(diff)

    numOfDifferences = max(diffList)
    if numOfDifferences == 0:
        pass
    else:
        dataSet_diff = dataSet.diff(periods=numOfDifferences)
        dataSet_diff.dropna(inplace=True)  # Drop null values
    print("After taking the differences", numOfDifferences, diffList)
    return dataSet_diff, numOfDifferences

def findBestParameters(train, train_diff, test, columnList):
    pq = []
    for name, column in train_diff[columnList].iteritems():
        print(f'Searching order of p and q for : {name}')
        stepwise_model = auto_arima(train_diff[name],start_p=1, start_q=1,max_p=7, max_q=7, seasonal=False,
            trace=True,error_action='ignore',suppress_warnings=True, stepwise=True,maxiter=1000)
        parameter = stepwise_model.get_params().get('order')
        print(f'optimal order for:{name} is: {parameter} \n\n')
        pq.append(stepwise_model.get_params().get('order'))
    print("Best parameters : ", pq)

    df_results_moni = pd.DataFrame(columns=['p', 'q', 'RMSE Latency', 'RMSE Throughput', 'RMSE CPU', 'RMSE Memory'])
    print('Grid Search Started')
    start = timer()
    for i in pq:
        if i[0] == 0 and i[2] == 0:
            pass
        else:
            print(f' Running for {i}')
            model = VARMAX(train_diff[columnList], order=(i[0], i[2])).fit(disp=False)
            result = model.forecast(steps=30)
            inv_res = inverse_diff(pd.concat([train, test], ignore_index=False, axis=0)[columnList], result)
            print(inv_res)
            Latencyrmse = np.sqrt(metrics.mean_squared_error(test[columnList[0]], inv_res.in_avg_response_time_1st_inv_diff))
            Throughputrmse = np.sqrt(metrics.mean_squared_error(test[columnList[1]], inv_res.in_throughput_1st_inv_diff))
            CPUrmse = np.sqrt(metrics.mean_squared_error(test[columnList[2]], inv_res.cpu_1st_inv_diff))
            Memoryrmse = np.sqrt(metrics.mean_squared_error(test[columnList[3]], inv_res.memory_1st_inv_diff))
            df_results_moni = df_results_moni.append(
                {'p': i[0], 'q': i[2], 'RMSE Latency': Latencyrmse, 'RMSE Throughput': Throughputrmse, 'RMSE CPU': CPUrmse,
                 'RMSE Memory': Memoryrmse}, ignore_index=True)
    end = timer()
    print(f' Total time taken to complete grid search in seconds: {(end - start)}')
    print(df_results_moni.sort_values(by=['RMSE Latency', 'RMSE Throughput', 'RMSE CPU', 'RMSE Memory']))

    x = []
    for i in range(len(df_results_moni)):
        x.append(df_results_moni.iloc[i,2]+df_results_moni.iloc[i,3]+df_results_moni.iloc[i,4]+df_results_moni.iloc[i,5])
    # print(x)
    p = int(df_results_moni.iloc[x.index(min(x)), 0])
    q = int(df_results_moni.iloc[x.index(min(x)), 1])
    return p,q


# invert difference
# invert difference
def invert_difference(orig_data, diff_data, interval):
	print(orig_data.iloc[2])


# Take inverse difference
def inverse_diff(actual_df, pred_df):
    df_res = pred_df.copy()
    columns = actual_df.columns
    for col in columns:
        df_res[str(col)+'_1st_inv_diff'] = actual_df[col].iloc[-1] + df_res[str(col)].cumsum()
    return df_res

def trainModel(train, test, numOfSteps, immediate1stPredictionsLatency, immediate1stPredictionsThroughput, immediate1stPredictionsCPU, immediate1stPredictionsMemory,
               dataSetID, columnList, test_v1):

    train_diff, numOfDifferences = makeStationary(train, columnList)
    print("After calling makeStationary, number of differences :- ", numOfDifferences)
    print(train_diff.info())

    p, q = findBestParameters(train, train_diff, test[:30], columnList)
    print(p, q, len(train_diff))

    dataset = pd.concat([train, test], ignore_index=False, axis=0)

    dataSet_diff = dataset.diff(periods=numOfDifferences)
    dataSet_diff.dropna(inplace=True)  # Drop null values
    # test_diff = dataSet_diff[dataSet_diff.loc[test.iloc[0]].index:]
    # print(dataSet_diff)
    print(str(test.index[0])[:-6])
    print(dataSet_diff.loc[str(test.index[0])[:-6]])
    for i in range(len(dataSet_diff)):
        if str(test.index[0])[:-6] == str(dataSet_diff.index[i])[:-6]:
            test_diff = dataSet_diff[i:]
            break
    print(test)
    print(test_diff)

    result = VARMAX(train_diff[columnList], order=(p, q)).fit(disp=False)
    print(result.predict(1, len(train)))
    print(result.predict(1, len(train))[columnList[0]])

    for i in range(len(columnList)):
        train_diff[columnList[i]].plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
                   title='ARIMA - Train data and fitted model')
        plt.plot(train_diff.index, result.predict(1, len(train_diff))[columnList[i]], label='Fitted model')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '. ' + 'Train data and fitted model '+ columnList[i] + '.png', dpi=300, bbox_inches='tight')
        plt.show()

    models = []
    train1 = train_diff.copy()
    for i in range(0, len(test_diff)):
        # t = time.time()
        result = VARMAX(train1[columnList], order=(p, q)).fit(disp=False)
        pred = result.get_forecast(steps=numOfSteps)
        # print(time.time() - t)
        print('training ',i)
        if i == 0 or i == 1 or i == 2:
            models.append(result)
        print(pred.predicted_mean.values[0][0], type(pred.predicted_mean.values[0][0]))
        immediate1stPredictionsLatency.append(pred.predicted_mean.values[0][0])
        immediate1stPredictionsThroughput.append(pred.predicted_mean.values[0][1])
        immediate1stPredictionsCPU.append(pred.predicted_mean.values[0][2])
        immediate1stPredictionsMemory.append(pred.predicted_mean.values[0][3])

        train1 = pd.concat([train1, test_diff[i:i + 1]], ignore_index=False, axis=0)

    return models, test_diff, dataSet_diff, train_diff

def plotResults(dataSet, trainSet, testSet, model, title1, title2, splitLineValue, dataSetID, columnList):
    #####################################
    # immediatePredictions
    #####################################
    # Dataset id is used when we loop through lots of datasets, coming from main file
    # Plot whole dataset, fitted values and immediate observations
    startDate = '2021-02-17 12:18'
    endDate = '2021-02-17 12:40'

    columnNameList = ['predictionsLatency', 'predictionsThroughput', 'predictionsCPU', 'predictionsMemory']
    for i in range(len(columnList)):
        dataSet[columnList[i]].plot(figsize=(10, 7), xlabel='Time', ylabel=columnList[i], label=columnList[i], title=title1, xlim=[str(trainSet.index[-13])[:-9], str(testSet.index[int((len(testSet)/6)*5)])[:-9]])
        plt.plot(testSet.index, testSet[columnNameList[i]], label='Predictions')
        # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
        plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '. ' + title1 + columnList[i] + '.png', dpi=300, bbox_inches='tight')
        plt.show()
        print(i)

        dataSet[columnList[i]].plot(figsize=(10, 7), xlabel='Time', ylabel=columnList[i], label=columnList[i],
                                    title=title1, xlim=[str(trainSet.index[-11])[:-9], str(testSet.index[int(len(testSet)/3)])[:-9]])
        plt.plot(testSet.index, testSet[columnNameList[i]], label='Predictions')
        # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
        plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '. ' + title2 + columnList[i] + '.png', dpi=300, bbox_inches='tight')
        plt.show()

        dataSet[columnList[i]].plot(figsize=(10, 7), xlabel='Time', ylabel=columnList[i], label=columnList[i],
                                    title=title1, xlim=[str(trainSet.index[-9])[:-9], str(testSet.index[int(len(testSet)/6)])[:-9]])
        plt.plot(testSet.index, testSet[columnNameList[i]], label='Predictions')
        # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
        plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
        plt.legend(loc='best')
        plt.savefig(str(dataSetID) + '. ' + title2 + columnList[i] + '(Much smaller).png', dpi=300, bbox_inches='tight')
        plt.show()

    # # Plot whole dataset, fitted values and immediate observations in small time frame
    # dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #              title=title2,
    #              xlim=[str(trainSet.index[-11])[:-9], str(testSet.index[int(len(testSet)/3)])[:-9]])
    # plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    # plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    # plt.legend(loc='best')
    # plt.savefig(str(dataSetID) + '. ' + title2 + '.png', dpi=300, bbox_inches='tight')
    # plt.show()
    #
    # #
    # # Plot whole dataset, fitted values and immediate observations in small time frame
    # dataSet.plot(figsize=(10, 7), xlabel='Time', ylabel='Latency', label='Latency',
    #              title=title2,
    #              xlim=[str(trainSet.index[-9])[:-9], str(testSet.index[int(len(testSet)/6)])[:-9]])
    # plt.axvline(pd.to_datetime(splitLineValue), color='k', linestyle='--')
    # plt.plot(testSet.index, testSet['predictions'], label='Predictions')
    # plt.plot(trainSet.index, model.fittedvalues, label='Fitted model')
    # plt.legend(loc='best')
    # plt.savefig(str(dataSetID) + '. ' + title2 + ' (Much smaller).png', dpi=300, bbox_inches='tight')
    # plt.show()