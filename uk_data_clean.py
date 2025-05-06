import pandas as pd

# === Load Data ===
gdp_df = pd.read_csv('data/uk-gdp.csv')  # expects columns like ['Quarter', 'GDP']
cpi_df = pd.read_csv('data/uk-cpi.csv')  # expects columns like ['Quarter', 'CPI_annual_rate']

# === Convert "YYYY Qx" to datetime (end of quarter) ===
def parse_quarter_string(qstr):
    year, q = qstr.split()
    quarter_end = {'Q1': '-03-31', 'Q2': '-06-30', 'Q3': '-09-30', 'Q4': '-12-31'}
    return pd.to_datetime(year + quarter_end[q])


gdp_df['DATE'] = gdp_df['Quarter'].apply(parse_quarter_string)
cpi_df['DATE'] = cpi_df['Quarter'].apply(parse_quarter_string)

# === Prepare GDP Data ===
# GDP is already in percent change format
gdp_df.rename(columns={'GDP': 'GDP_pct'}, inplace=True)
gdp_df = gdp_df[['DATE', 'GDP_pct']]

# === Process CPI: convert annual rate to quarterly rate, build index, then compute QoQ % change ===
# Convert annual inflation rate (e.g., 5.8%) to decimal quarterly rate
cpi_df['quarterly_rate'] = (1 + cpi_df['CPI'] / 100) ** (1/4) - 1

# Reconstruct a synthetic CPI index (base 100)
cpi_df['CPI_index'] = 100 * (1 + cpi_df['quarterly_rate']).cumprod()

# Calculate QoQ percent change in index
cpi_df['CPI_pct'] = cpi_df['CPI_index'].pct_change() * 100

# Round to 3 decimal places
cpi_df['CPI_pct'] = cpi_df['CPI_pct'].round(3)

# Drop first row (NaN in pct_change)
cpi_df = cpi_df[['DATE', 'CPI_pct']].dropna()

# === Merge GDP and CPI ===
uk_df = pd.merge(gdp_df, cpi_df, on='DATE', how='inner')

# === Export Cleaned Data ===
uk_df.to_csv('data/uk_macro_cleaned.csv', index=False)

# === (Optional) Preview the result ===
print(uk_df.head())
