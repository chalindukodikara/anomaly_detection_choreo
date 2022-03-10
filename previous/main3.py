######################################################################
######################Air Passengers Dataset##########################
######################################################################

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

#Import csv data
df=pd.read_csv('AirPassengers.csv')

#We need to set the Month column as index and convert it into datetime
df.set_index('Year', inplace=True) #inplace = If True, modifies the DataFrame in place (do not create a new object)
df.index = pd.to_datetime(df.index) #Convert argument to datetime.
df.head()

#Split the dataset into test and train
X = df
size = int(len(X) * 0.7)
train, test = X[0:size], X[size:len(X)]
print("Training set size :", len(train))
print("Test set size :", len(test))

#Plot original data
train.plot(xlabel = 'Year', ylabel = 'Number of Passengers', title = 'Original Series')
print(train)
plt.show()

#Check for stationary
result = adfuller(train['NumOfPassengers'])
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Decompose data into trends and seasonal
decomposition = seasonal_decompose(train, model = 'multiplicative')

trend = decomposition.trend
seasonal = decomposition.seasonal
residual = decomposition.resid
plt.figure(figsize=(16,7))
fig = plt.figure(1)

plt.title('View trend and seasonality')
plt.subplot(411)
plt.plot(train, label='Original')
plt.legend(loc='best')
plt.subplot(412)
plt.plot(trend, label='Trend')
plt.legend(loc='best')
plt.subplot(413)
plt.plot(seasonal,label='Seasonality')
plt.legend(loc='best')
plt.subplot(414)
plt.plot(residual, label='Residuals')
plt.legend(loc='best')
plt.show()


## Adf Test, Dickey–Fuller test
print('adf test results : ', ndiffs(train, test='adf'))
# KPSS test, Kwiatkowski–Phillips–Schmidt–Shin (KPSS) test
print('kpss test results : ', ndiffs(train, test='kpss'))
# PP test, Phillips–Perron test is a unit root test
print('pp test results : ', ndiffs(train, test='pp'))

#Plot acf & pacf for original data
plot_acf(train)
plot_pacf(train)
plt.show()

#Take first difference
train['1difference']=train['NumOfPassengers']-train['NumOfPassengers'].shift(1)
train['1difference'].plot(title = 'After first difference')
plt.show()

#Check for stationary after first difference
result = adfuller(train['1difference'].dropna())
print('ADF Statistic for first difference: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

#Take second difference
train['2difference']=train['1difference']-train['1difference'].shift(1)
train['2difference'].plot(title = 'After second difference')
plt.show()

#Check for stationary after second difference
result = adfuller(train['2difference'].dropna())
print('ADF Statistic for second difference: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

plot_acf(train['2difference'].dropna(), title = 'ACF Diagram after 2nd difference')
plot_pacf(train['2difference'].dropna(), title = 'PACF Diagram after 2nd difference')
plt.show()

#Seasonal difference check
train['Seasonal_Difference']=train['NumOfPassengers']-train['NumOfPassengers'].shift(12)
train['Seasonal_Difference'].plot(title='Series after seasonal difference')
plt.show()

#Check for stationary
result = adfuller(train['Seasonal_Difference'].dropna())
print('ADF Statistic for seasonal stationary check: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))

# train['1difference']=train['1difference']-train['1difference'].shift(1)
# train['1difference'].plot(title = 'After second difference')


#Drop extra columns added above
train = train.drop(columns="1difference")
train = train.drop(columns="2difference")
train = train.drop(columns="Seasonal_Difference")

# print(train)

#Create the model and fit it
model = ARIMA(train['NumOfPassengers'], order=(18,2,1))
# model=SARIMAX(train['NumOfPassengers'],order=(4,2,1), seasonal_order=(1, 0, 0, 12))
result = model.fit()

#Print summary of the model
print(result.summary())

#plot the residuals of the model. The residuals are the difference between the original values and the predicted values from the model.
result.resid.plot(kind='kde', title = 'residuals of the model')
plt.show()
testSet = test[:]
# test['predictions'] = np.nan
df2 = pd.concat([train, test])
prediction = result.predict(start=100,end=143)
df2['predictions'] = prediction
# print(df2['predictions'])

testValues = []
for i in range(len(testSet.values.tolist())):
	testValues.append(testSet.values.tolist()[i][0])

def findAnomalies(predictionValues, testValues):
	squaredErrors = []
	# print('xxxxxxxxxxxxxxxxxxxxxxxxxxx', len(predictionValues), len(testValues))
	for i in range(len(testValues)):
		squaredErrors.append((predictionValues[i] - testValues[i]) ** 2)
	anomaly = (squaredErrors >= threshold).astype(int)
	return anomaly
squared_errors = result.resid.values ** 2
threshold = np.mean(squared_errors) + np.std(squared_errors)

anomaly = findAnomalies(prediction.values.tolist(), testValues)
rmse = sqrt(mean_squared_error(prediction.values.tolist(), testValues))
print('Test RMSE: %.3f' % rmse)

test['predictions'] = prediction
test['Anomaly'] = anomaly
print(test['Anomaly'][0])
# print(test)

testAnomaly = test[:]
while True:
	x = 0
	y = len(testAnomaly)
	for i in range(len(testAnomaly)):
		if testAnomaly['Anomaly'][i] == 0:
			testAnomaly = testAnomaly.drop(testAnomaly.head(len(testAnomaly)).index[i])
			break
		if i == y-1:
			x = 1
	if x == 1:
		break


print(test)
print(testAnomaly)

df2[['NumOfPassengers','predictions']].plot(title = 'Predictions for test set values')
plt.scatter(testAnomaly.index, testAnomaly['NumOfPassengers'], marker = 'x', color = 'red',  label = 'Anomaly')
plt.legend()
plt.show()


# new_dates=[df.index[-1]+DateOffset(months=x) for x in range(1,48)]
# df_pred=pd.DataFrame(index=new_dates,columns =df.columns)
# df_pred.head()
#
# df3=pd.concat([df2, df_pred])
# df3['predictions']=result.predict(start=99,end=191)
# df3[['NumOfPassengers','predictions']].plot(title = 'Predictions for future values not in dataset')
# plt.show()

# print(df3)

# output = result.forecast()
# yhat = output[0]
# print(yhat)