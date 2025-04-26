import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

macro_data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)

# Define noise parameters
noise_std_gdp = 0.01 * macro_data['GDP'].std()  # 1% of GDP standard deviation
noise_std_cpi = 0.01 * macro_data['CPI'].std()  # 1% of CPI standard deviation

# Add Gaussian noise
macro_data_noisy = macro_data.copy()
macro_data_noisy['GDP'] += np.random.normal(0, noise_std_gdp, size=len(macro_data))
macro_data_noisy['CPI'] += np.random.normal(0, noise_std_cpi, size=len(macro_data))

macro_data_noisy.to_csv('macro_data_noisy.csv')

# Now you have the noisy version!
print(macro_data_noisy.head())
