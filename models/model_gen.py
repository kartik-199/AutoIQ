# Data manipulation
from typing import final

import numpy as np
import pandas as pd
from pandas.core.common import random_state

# ML Modeling Tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib

# Dataset from UC Irvine Repo
from ucimlrepo import fetch_ucirepo

# Fetch the automobile MPG dataset
auto_mpg = fetch_ucirepo(id=9)

# Convert to dataframe & squeeze dimensionality to save memory
features_df = pd.DataFrame(auto_mpg.data.features)
target = auto_mpg.data.targets.squeeze()
print(features_df.columns)

# ------ Feature engineering ------

# Create interaction terms
features_df['weight_acceleration'] = features_df['weight'] * features_df['acceleration']
features_df['cylinders_displacement'] = features_df['cylinders'] * features_df['displacement']

# Power transformations - apply mathematical transformations to features
features_df['log_weight'] = np.log1p(features_df['weight'])
features_df['sqrt_displacement'] = np.sqrt(features_df['displacement'])

# Categorical encoding
features_df['origin_1'] = (features_df['origin'] == 1).astype('int')
features_df['origin_2'] = (features_df['origin'] == 2).astype('int')
features_df['origin_3'] = (features_df['origin'] == 3).astype('int')

# Bin continuous variable - horsepower
features_df['horsepower_bin'] = pd.cut(features_df['horsepower'], bins=4, labels=False)

# Generate polynomial transformation based features
poly = PolynomialFeatures(degree=2, include_bias=False)
poly_features = poly.fit_transform(features_df[['weight', 'acceleration', 'displacement']])
poly_features_names = poly.get_feature_names_out(['weight', 'acceleration', 'displacement'])

# Create respective df
poly_df = pd.DataFrame(poly_features, columns = poly_features_names)

# Concatenate poly_df + features_df
final_features = pd.concat([features_df, poly_df], axis=1)

# Handle missing values
imputer = SimpleImputer(strategy="mean")
final_features_imputed = imputer.fit_transform(final_features)


# ------ Model training ------

# Divide into training and testing splits
x_train, x_test, y_train, y_test = train_test_split(
    final_features_imputed,
    target,
    test_size=0.15,
    random_state=42
)

# Scale the features to have similar range
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.fit_transform(x_test)

# Train the multiple regression model
lr_model = LinearRegression()
lr_model.fit(x_train_scaled, y_train)

# Make prediction set
y_prediction = lr_model.predict(x_test_scaled)

# Calculate performance metrics
print("Mean Squared Error:", mean_squared_error(y_test, y_prediction))
print("R2 Score:", r2_score(y_test, y_prediction))

# Save model
joblib.dump(lr_model, "multiple_regression_model_mpg_engineered.pkl")

# Feature importance analysis
feature_importance = pd.DataFrame({
    'feature': final_features.columns,
    'importance': np.abs(lr_model.coef_)
})
print("\nTop 10 Most Important Features:")
print(feature_importance.sort_values('importance', ascending=False).head(10))