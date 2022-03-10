######################################################################
############# Airline Passengers Dataset - Exponential Smoothing #####
######################################################################
# dataframe opertations - pandas
import pandas as pd
# plotting data - matplotlib
from matplotlib import pyplot as plt
# time series - statsmodels
# Seasonality decomposition
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.seasonal import seasonal_decompose
# holt winters
# single exponential smoothing
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# double and triple exponential smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing


airline = pd.read_csv('airline_passengers.csv',index_col='Month', parse_dates=True)
airline = pd.read_csv('airline_passengers.csv',index_col='Month', parse_dates=True)

# finding shape of the dataframe
print(airline.shape)
# having a look at the data
print(airline.head())
# plotting the original data
airline[['Thousands of Passengers']].plot(title='Passengers Data')

decompose_result = seasonal_decompose(airline['Thousands of Passengers'], model='multiplicative')
decompose_result.plot();

# Set the frequency of the date time index as Monthly start as indicated by the data
airline.index.freq = 'MS'
# Set the value of Alpha and define m (Time Period)
m = 12
alpha = 1/(2*m)

airline['HWES1'] = SimpleExpSmoothing(airline['Thousands of Passengers']).fit(smoothing_level=alpha,optimized=False,use_brute=True).fittedvalues
airline[['Thousands of Passengers','HWES1']].plot(title='Holt Winters Single Exponential Smoothing');
plt.show()

airline['HWES2_ADD'] = ExponentialSmoothing(airline['Thousands of Passengers'],trend='add').fit().fittedvalues
airline['HWES2_MUL'] = ExponentialSmoothing(airline['Thousands of Passengers'],trend='mul').fit().fittedvalues
airline[['Thousands of Passengers','HWES2_ADD','HWES2_MUL']].plot(title='Holt Winters Double Exponential Smoothing: Additive and Multiplicative Trend');
# plt.show()

airline['HWES3_ADD'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='add', seasonal='add',seasonal_periods=12).fit().fittedvalues
airline['HWES3_MUL'] = ExponentialSmoothing(airline['Thousands of Passengers'], trend='mul', seasonal='mul',seasonal_periods=12).fit().fittedvalues
airline[['Thousands of Passengers','HWES3_ADD', 'HWES3_MUL']].plot(title='Holt Winters Triple Exponential Smoothing: Additive and Multiplicative Seasonality');

plt.show()


forecast_data = pd.read_csv('airline_passengers.csv',index_col='Month',parse_dates=True)
# forecast_data.index.freq = 'MS'
# Split into train and test set
train_airline = forecast_data[:120]
test_airline = forecast_data[120:]

fitted_model = ExponentialSmoothing(train_airline['Thousands of Passengers'],trend='mul',seasonal='mul',seasonal_periods=12).fit()
test_predictions = fitted_model.forecast(24)
train_airline['Thousands of Passengers'].plot(legend=True,label='TRAIN')
test_airline['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(6,4))
test_predictions.plot(legend=True,label='PREDICTION')
plt.title('Train, Test and Predicted Test using Holt Winters')
plt.show()


test_airline['Thousands of Passengers'].plot(legend=True,label='TEST',figsize=(9,6))
test_predictions.plot(legend=True,label='PREDICTION', xlim=['1959','1961']);
plt.show()

from sklearn.metrics import mean_absolute_error,mean_squared_error
print(f'Mean Absolute Error = {mean_absolute_error(test_airline,test_predictions)}')
print(f'Mean Squared Error = {mean_squared_error(test_airline,test_predictions)}')
