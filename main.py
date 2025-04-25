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

# Display heads
print(gdp.head())
print(inflation.head())

# Save locally if you want
gdp.to_csv('gdp_data.csv')
inflation.to_csv('inflation_data.csv')
