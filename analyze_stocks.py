import pandas as pd
import matplotlib.pyplot as plt
import datetime
import numpy as np
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('stock-data.csv', parse_dates=[0])
indexed = data.set_index(['Unnamed: 0'])

# Company Stocks
apple = pd.DataFrame(data=indexed['AAPL'])
amazon = pd.DataFrame(data=indexed['AMZN'])
google = pd.DataFrame(data=indexed['GOOGL'])
microsoft = pd.DataFrame(data=indexed['MSFT'])

# Forecast Data
forecast = 30  # 30 days ahead

apple['AAPL forecasted'] = apple['AAPL'].shift(-forecast)
amazon['AMZN forecasted'] = amazon['AMZN'].shift(-forecast)
google['GOOGL forecasted'] = indexed['GOOGL'].shift(-forecast)
microsoft['MSFT forecasted'] = microsoft['MSFT'].shift(-forecast)


# Train, Test, Predict Apple Stock Prices
X_apple = np.array(apple.drop(['AAPL forecasted'], 1))
X_apple = preprocessing.scale(X_apple)
X_apple_lately = X_apple[-forecast:]
X_apple = X_apple[:-forecast]

apple.dropna(inplace=True)
y_apple = np.array(apple['AAPL forecasted'])

X_train, X_test, y_train, y_test = train_test_split(X_apple, y_apple)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

forecastAppleStocks = clf.predict(X_apple_lately)
print("Forecast of apple's stocks: ", forecastAppleStocks)


# Train, Test, Predict Amazon Stock Prices
X_amazon = np.array(amazon.drop(['AMZN forecasted'], 1))
X_amazon = preprocessing.scale(X_amazon)
X_amazon_lately = X_amazon[-forecast:]
X_amazon = X_amazon[:-forecast]

amazon.dropna(inplace=True)
y_amazon = np.array(amazon['AMZN forecasted'])

X_train, X_test, y_train, y_test = train_test_split(X_amazon, y_amazon)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

forecastAmazonStocks = clf.predict(X_amazon_lately)
print("Forecast of amazon's stocks: ", forecastAmazonStocks)


# Train, Test, Predict Google Stock Prices
X_google = np.array(google.drop(['GOOGL forecasted'], 1))
X_google = preprocessing.scale(X_google)
X_google_lately = X_google[-forecast:]
X_google = X_google[:-forecast]

google.dropna(inplace=True)
y_google = np.array(google['GOOGL forecasted'])

X_train, X_test, y_train, y_test = train_test_split(X_google, y_google)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

forecastGoogleStocks = clf.predict(X_google_lately)
print("Forecast of google's stocks: ", forecastGoogleStocks)


# Train, Test, Predict Microsoft Stock Prices
X_microsoft = np.array(microsoft.drop(['MSFT forecasted'], 1))
X_microsoft = preprocessing.scale(X_microsoft)
X_microsoft_lately = X_microsoft[-forecast:]
X_microsoft = X_microsoft[:-forecast]

microsoft.dropna(inplace=True)
y_microsoft = np.array(microsoft['MSFT forecasted'])

X_train, X_test, y_train, y_test = train_test_split(X_microsoft, y_microsoft)
clf = LinearRegression()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print("Accuracy: ", accuracy)

forecastMicrosoftStocks = clf.predict(X_microsoft_lately)
print("Forecast of microsoft's stocks: ", forecastMicrosoftStocks)


# Create dates for forecasted stocks 30 days ahead
today = datetime.date.today()
tomorrow = (today + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
dates = pd.date_range(tomorrow, periods=30, freq='D')

# Create dataframe for predicted apple stocks
applePredict = pd.DataFrame(data=forecastAppleStocks, index=dates)
applePredict.columns = ['AAPL Forecasted']

# Create dataframe for predicted amazon stocks
amazonPredict = pd.DataFrame(data=forecastAmazonStocks, index=dates)
amazonPredict.columns = ['AMZN Forecasted']

# Create dataframe for predicted google stocks
googlePredict = pd.DataFrame(data=forecastGoogleStocks, index=dates)
googlePredict.columns = ['GOOGL Forecasted']

# Create dataframe for predicted microsoft stocks
microsoftPredict = pd.DataFrame(data=forecastMicrosoftStocks, index=dates)
microsoftPredict.columns = ['MSFT Forecasted']



# Graph current and forecasted Apple stock
plt.plot(indexed.index, indexed['AAPL'], color="blue")
plt.plot(applePredict.index, applePredict['AAPL Forecasted'], color="red")
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()

# Graph current and forecasted Amazon stock
plt.plot(indexed.index, indexed['AMZN'], color="green")
plt.plot(amazonPredict.index, amazonPredict['AMZN Forecasted'], color="yellow")
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()

# Graph current and forecasted Google stock
plt.plot(indexed.index, indexed['GOOGL'], color="brown")
plt.plot(googlePredict.index, googlePredict['GOOGL Forecasted'], color="pink")
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.legend()

# Graph current and forecasted Microsoft stock
plt.plot(indexed.index, indexed['MSFT'], color="orange")
plt.plot(microsoftPredict.index, microsoftPredict['MSFT Forecasted'], color="black")
plt.xlabel('Date')
plt.ylabel('Closing Price ($)')
plt.title('Stock Market Price')
plt.legend()

plt.savefig('stock-graph.png')
plt.show()
