import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def create_supervised(df, n_lags=4):
    df = df.copy().sort_values('Date')
    for i in range(1, n_lags + 1):
        df[f'GDP_lag_{i}'] = df['GDP_pct'].shift(i)
        df[f'CPI_lag_{i}'] = df['CPI_pct'].shift(i)
    df['Target'] = df['GDP_pct'].shift(-1)
    df.dropna(inplace=True)
    return df


def split_data(df):
    X = df.drop(columns=['Date', 'Country', 'GDP_pct', 'CPI_pct', 'Target'])
    y = df['Target']
    split_idx = int(len(X) * 0.8)
    return X.iloc[:split_idx], X.iloc[split_idx:], y.iloc[:split_idx], y.iloc[split_idx:],


supervised_base_df = create_supervised(pd.read_csv('data/dataset-base.csv'), n_lags=4)
supervised_noisy_df = create_supervised(pd.read_csv('data/dataset-noisy.csv'), n_lags=4)

X_base_train, X_base_test, y_base_train, y_base_test = split_data(supervised_base_df)
X_noisy_train, X_noisy_test, y_noisy_train, y_noisy_test = split_data(supervised_noisy_df)

model_base = RandomForestRegressor(random_state=42)
model_base.fit(X_base_train, y_base_train)
y_base_pred = model_base.predict(X_base_test)

model_noisy = RandomForestRegressor(random_state=42)
model_noisy.fit(X_noisy_train, y_noisy_train)
y_noisy_pred = model_noisy.predict(X_noisy_test)

mse_base = mean_squared_error(y_base_test, y_base_pred)
mse_noisy = mean_squared_error(y_noisy_test, y_noisy_pred)

mae_base = mean_absolute_error(y_base_test, y_base_pred)
mae_noisy = mean_absolute_error(y_noisy_test, y_noisy_pred)

r2_base = r2_score(y_base_test, y_base_pred)
r2_noisy = r2_score(y_noisy_test, y_noisy_pred)

print(f"Base Data \n MSE: {mse_base:.4f}, MAE: {mae_base:.4f}, R^2: {r2_base:.4f}")
print(f"Noisy Data \n MSE: {mse_noisy:.4f}, MAE: {mae_noisy:.4f}, R^2: {r2_noisy:.4f}")
