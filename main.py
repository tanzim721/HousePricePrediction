import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Load dataset
data = pd.read_csv("train.csv")

# Select features
features = [
    'GrLivArea',
    'BedroomAbvGr',
    'FullBath',
    'GarageArea',
    'YearBuilt',
    'TotRmsAbvGrd',
    'OverallQual',
    'LotArea'
]

X = data[features]
y = data['SalePrice']

# Handle missing values
X = X.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Create model
model = LinearRegression()

# Train model
model.fit(X_train, y_train)

# Predict
predictions = model.predict(X_test)

# Evaluate
error = mean_absolute_error(y_test, predictions)

print("Error:", error)

# Compare
comparison = pd.DataFrame({
    "Actual": y_test,
    "Predicted": predictions
})

print(comparison.head())

# Visualization
plt.scatter(data['GrLivArea'], data['SalePrice'])

plt.xlabel("Living Area")
plt.ylabel("Sale Price")

plt.show()

print(data[features + ['SalePrice']].corr()['SalePrice'])