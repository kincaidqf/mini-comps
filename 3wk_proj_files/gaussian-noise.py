import numpy as np
import pandas as pd

# Set the random seed for reproducibility (allows 'random' results to be generated from same seed for consistency)
np.random.seed(42)

uk_df = pd.read_csv('data/uk_macro_cleaned.csv')
us_df = pd.read_csv('data/us_macro_cleaned.csv')

# Add country label
us_df['Country'] = 'US'
uk_df['Country'] = 'UK'

# Add US data to the bottom of UK data (vertical concat)
combined_df = pd.concat([uk_df, us_df], ignore_index=True)

# Add Gaussian noise to dataframe with standard distribution 0.1
def add_guassian_noise(df, std=0.1, seed=42):
    np.random.seed(seed)
    noisy_df = df.copy()
    noisy_df['GDP_pct'] += np.random.normal(0, std, size=len(df))
    noisy_df['CPI_pct'] += np.random.normal(0, std, size=len(df))
    return noisy_df


noisy_df = add_guassian_noise(combined_df, std=0.1)

# Round 3 sig fig
noisy_df['GDP_pct'] = noisy_df['GDP_pct'].round(3)
noisy_df['CPI_pct'] = noisy_df['CPI_pct'].round(3)

# Save clean and noisy datasets to csv
combined_df.to_csv('data/dataset-base.csv')
noisy_df.to_csv('data/dataset-noisy.csv')
