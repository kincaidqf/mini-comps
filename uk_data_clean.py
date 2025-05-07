import pandas as pd

# Load gdp and cpi data from csv
gdp_df = pd.read_csv('data/uk-gdp.csv')
cpi_df = pd.read_csv('data/uk-cpi.csv')

# Convert date format ("YYYY Qx") to datetime (end of quarter)
def parse_quarter_string(qstr):
    year, q = qstr.split()
    quarter_end = {'Q1': '-03-31', 'Q2': '-06-30', 'Q3': '-09-30', 'Q4': '-12-31'}
    return pd.to_datetime(year + quarter_end[q])


gdp_df['DATE'] = gdp_df['Quarter'].apply(parse_quarter_string)
cpi_df['DATE'] = cpi_df['Quarter'].apply(parse_quarter_string)

# GDP is already in percent change format, rename column for clarity
gdp_df.rename(columns={'GDP': 'GDP_pct'}, inplace=True)
gdp_df = gdp_df[['DATE', 'GDP_pct']]

# Convert annual inflation rate (e.g., 5.8%) to decimal quarterly rate
# uses formula CPIq = ((1+CPIa)/100)^1/4 - 1, CPIq = quarterly, CPIa = annual
cpi_df['quarterly_rate'] = (1 + cpi_df['CPI'] / 100) ** (1/4) - 1

# Treat CPI as index with a base of 100
cpi_df['CPI_index'] = 100 * (1 + cpi_df['quarterly_rate']).cumprod()

# Calculate QoQ percent change in CPI index
cpi_df['CPI_pct'] = cpi_df['CPI_index'].pct_change() * 100

# Round to 3 decimal places
cpi_df['CPI_pct'] = cpi_df['CPI_pct'].round(3)

# Drop first row (NaN in pct_change)
cpi_df = cpi_df[['DATE', 'CPI_pct']].dropna()

# Merge GDP and CPI dataframes into one dataset
uk_df = pd.merge(gdp_df, cpi_df, on='DATE', how='inner')

# Save cleaned data to csv file
uk_df.to_csv('data/uk_macro_cleaned.csv', index=False)

# Preview dataset
print(uk_df.head())
