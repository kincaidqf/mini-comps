import numpy as np
import pandas as pd

# Set the random seed for reproducibility
np.random.seed(42)

uk_df = pd.read_csv('data/uk_macro_cleaned.csv')
us_df = pd.read_csv('data/us_macro_cleaned.csv')

combined_df = pd.concat([uk_df, us_df], ignore_index=True)

print(combined_df.head())
