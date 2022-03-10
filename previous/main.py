######################################################################
######################Shampoo Sales Dataset###########################
######################################################################

# evaluate an ARIMA model using a walk-forward validation
from pandas import read_csv, Series
from datetime import datetime
import time
from matplotlib import pyplot
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf
from statsmodels.graphics.tsaplots import plot_acf
from pandas.plotting import autocorrelation_plot
from sklearn.metrics import mean_squared_error
from math import sqrt
import numpy as np

# load dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('previous\shampoo-sales.csv', header=0, index_col=0, parse_dates=True, squeeze=True, date_parser=parser)
series.index = series.index.to_period('M')

series.plot(xlabel = 'Year', ylabel = 'Number of Sales', title = 'Original Series')
pyplot.show()

plot_pacf(series)
plot_acf(series)
pyplot.show()
# autocorrelation_plot(series)
pyplot.show()

# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)

# series.plot()
# pyplot.show()

diff = difference(series)

# pyplot.plot(diff)
# pyplot.show()

plot_pacf(diff, title='Partial Autocorrelation after 1st difference')
plot_acf(diff, title='Autocorrelation after 1st difference')
pyplot.show()
# autocorrelation_plot(diff)
pyplot.show()

# split into train and test sets
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
print(train, test)
history = [x for x in train]
predictions = list()

def findAnomalies(squared_errors, k, yhat, obs):
	threshold = np.mean(squared_errors) + np.std(squared_errors)
	anomaly = (squared_errors >= threshold).astype(int)
	if k == True:
		x = (yhat - obs) ** 2
		if x >= threshold:
			np.append(anomaly, 1)
		else:
			# todo find a way to add items to ndarray
			anomaly = anomaly.tolist() #Making numpy ndarray to list
			anomaly.append(0)
	return anomaly, threshold

anomaly = []
# walk-forward validation
for t in range(len(test)):
	model = ARIMA(history, order=(5,1,1))
	l = time.time()
	print(history)
	print(type(history))
	model_fit = model.fit()
	x = time.time() - l
	# print(x)
	output = model_fit.forecast()
	yhat = output[0]
	predictions.append(yhat)
	obs = test[t]
	squared_error = model_fit.resid ** 2
	k = False
	if t == len(test)-1:
		k = True
	anomaly , threshold = findAnomalies(squared_error, k, yhat, obs)
	history.append(obs)
	print('predicted=%f, expected=%f' % (yhat, obs))


# evaluate forecasts
#The RMSE is the square root of the variance of the residuals
rmse = sqrt(mean_squared_error(test, predictions))
print('Test RMSE: %.3f' % rmse)
#
# # series['anomaly'] = anomaly
# print(type(series))

# plot forecasts against actual outcomes
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()
k = 0
for i in series.keys():
	print(i, end=" ")
	print(series[i], end =" ")
	print(anomaly[k])
	k += 1