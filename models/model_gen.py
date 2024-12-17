from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from ucimlrepo import fetch_ucirepo
import joblib

# Fetch the dataset from UCI machine learning repository
auto_mpg = fetch_ucirepo(id=9)

# Extract data
features = auto_mpg.data.features
target = auto_mpg.data.targets

# Handle missing values by imputing with mean
imputer = SimpleImputer(strategy="mean")
features_imputed = imputer.fit_transform(features)

# Split the dataset
x_train, x_test, y_train, y_test = train_test_split(features_imputed, target, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train the model
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train)

# Make predictions
y_prediction = lr_model.predict(x_test_scaled)

# Evaluate the model
print("Linear Regression")
print("Mean Squared Error:", mean_squared_error(y_test, y_prediction))
print("R2 Score:", r2_score(y_test, y_prediction))

joblib.dump(lr_model, 'multiple_regression_model_mpg.pkl')
print("Model saved successfully.")