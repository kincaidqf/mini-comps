import pandas as pd
import pandas_datareader.data as web
import datetime

# Set start and end dates for data
start = datetime.datetime(2000, 1, 1)  # You can adjust as needed
end = datetime.datetime(2024, 1, 1)

# GDP (Quarterly, Seasonally Adjusted Annual Rate)
gdp = web.DataReader('GDP', 'fred', start, end)

# Inflation - using Personal Consumption Expenditures (PCE) Price Index or CPI
# Common inflation proxy: CPIAUCSL = Consumer Price Index for All Urban Consumers
inflation = web.DataReader('CPIAUCSL', 'fred', start, end)

# Resample monthly CPI to quarterly CPI (mean of months in quarter)
inflation_quarterly = inflation.resample('Q').mean()

# After resampling CPI to quarterly
inflation_quarterly.index = inflation_quarterly.index + pd.Timedelta(days=1)

# Merge on index (date)
macro_data = pd.merge(gdp, inflation_quarterly, left_index=True, right_index=True)
macro_data.columns = ['GDP', 'CPI']

# Save locally
gdp.to_csv('gdp_data.csv')
inflation.to_csv('inflation_data.csv')
inflation_quarterly.to_csv('inflation_quarterly_data.csv')
macro_data.to_csv('macro_data.csv')

'''
For data visualization:

# Display heads
print(gdp.head())
print(inflation.head())
print(inflation_quarterly.head())
print(macro_data.head())

'''

