import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

macro_data = pd.read_csv('macro_data.csv', index_col=0, parse_dates=True)
macro_data_noisy = pd.read_csv('macro_data_noisy.csv', index_col=0, parse_dates=True)

# Create targets for gdp (if GDP_next > GDP then target = 1, else 0)
macro_data['GDP_next'] = macro_data['GDP'].shift(-1)
macro_data['Target'] = (macro_data['GDP_next'] > macro_data['GDP']).astype(int)

# Removing last row of table so that it doesn't try to learn from this row (since no next row available)
macro_data = macro_data.dropna()

X = macro_data[['GDP', 'CPI']]
y = macro_data['Target']

# Split data into train/test sets (70% train 30% test)
train_size = int(0.7 * len(macro_data))
X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]

# Random forest classifier model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions for test data
y_pred = model.predict(X_test)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")



