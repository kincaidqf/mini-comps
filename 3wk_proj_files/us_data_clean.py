import pandas as pd
import pandas_datareader.data as web
import datetime

# Set start and end dates for data
start = datetime.datetime(1989, 1, 1)  # You can adjust as needed
end = datetime.datetime(2024, 1, 1)

# GDP (Quarterly, Seasonally Adjusted Annual Rate)
gdp = web.DataReader('GDP', 'fred', start, end)

# Inflation - using Personal Consumption Expenditures (PCE) Price Index or CPI
# Common inflation proxy: CPIAUCSL = Consumer Price Index for All Urban Consumers
inflation = web.DataReader('CPIAUCSL', 'fred', start, end)

# Resample monthly CPI to quarterly CPI (mean of months in quarter)
inflation_quarterly = inflation.resample('Q').mean()

# After resampling CPI to quarterly
gdp.index = gdp.index + pd.Timedelta(days=-1)

# Merge on index (date)
macro_data = pd.merge(gdp, inflation_quarterly, left_index=True, right_index=True)
macro_data.columns = ['GDP_pct', 'CPI_pct']

# Convert to percent
macro_data['GDP_pct'] = macro_data['GDP_pct'].pct_change() * 100
macro_data['CPI_pct'] = macro_data['CPI_pct'].pct_change() * 100

# Round to 3 digits
macro_data['GDP_pct'] = macro_data['GDP_pct'].round(3)
macro_data['CPI_pct'] = macro_data['CPI_pct'].round(3)

# Drop empty first rows (na change from previous quarter)
macro_data.dropna(subset=['GDP_pct', 'CPI_pct'], inplace=True)

# Save to csv
macro_data.to_csv('data/us_macro_cleaned.csv')


