import pandas as pd
import datetime
from pandas_datareader import data


stocks = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
data_source = 'google'

today = datetime.date.today().strftime('%Y-%m-%d')

startDate = '2000-01-01'
endDate = today

# Retrieve data from google finance api
# Reference from : http://www.learndatasci.com/python-finance-part-yahoo-finance-api-pandas-matplotlib/
stockData = data.DataReader(stocks, data_source, startDate, endDate)
closeStock = stockData.ix['Close']


# stockData = stockData.to_frame()
# print(stockData)


allWeekdays = pd.date_range(start=startDate, end=endDate, freq='B')
closeStock = closeStock.reindex(allWeekdays)


print(closeStock)


# Clean up data (Filter out public holidays)
closeStock = closeStock.dropna(axis=0, how='any')
closeStock.to_csv('stock-data.csv')
print(closeStock)





