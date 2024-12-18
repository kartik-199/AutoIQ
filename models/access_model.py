import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


# Preprocessing pipeline
def preprocess_input(weight, acceleration, displacement, cylinders, horsepower, model_year, origin):
    input_dict = {
        'weight' : [weight],
        'acceleration' : [acceleration], 
        'displacement' : [displacement],
        'cylinders' : [cylinders],
        'horsepower' : [horsepower],
        'model_year' : [model_year],
        'origin' : [origin]
    }
    # Create a dataframe from the dictionary
    input_df = pd.DataFrame(input_dict)

    # Add the interaction terms
    input_df['weight_acceleration'] = input_df['weight'] * input_df['acceleration']
    input_df['cylinders_displacement'] = input_df['cylinders'] * input_df['displacement']

    # Apply power transformations
    input_df['log_weight'] = np.log1p(input_df['weight'])
    input_df['sqrt_displacement'] = np.sqrt(input_df['displacement'])

    # Categorical encoding for origin
    input_df['origin_1'] = (input_df['origin'] == 1).astype('int')
    input_df['origin_2'] = (input_df['origin'] == 2).astype('int')
    input_df['origin_3'] = (input_df['origin'] == 3).astype('int')

    # Bin continuous variable - horsepower
    input_df['horsepower_bin'] = pd.cut(input_df['horsepower'], bins=4, labels=False).astype('float')

    # Generate polynomial features
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_features = poly.fit_transform(input_df[['weight', 'acceleration', 'displacement']])
    poly_feature_names = poly.get_feature_names_out(['weight', 'acceleration', 'displacement'])
    poly_df = pd.DataFrame(poly_features, columns=poly_feature_names)

    # Combine polynomial features with the original DataFrame
    final_input_df = pd.concat([input_df, poly_df], axis=1)

    # Handle missing values (e.g., impute if necessary)
    final_input_df.fillna(final_input_df.mean(), inplace=True)
    return final_input_df

# Define the function to make predictions
def get_model_prediction(weight, acceleration, displacement, cylinders, horsepower, model_year, origin):
    lr_model = joblib.load('models/multiple_regression_model_mpg_engineered.pkl')

    # Preprocess the input
    preprocessed_input = preprocess_input(weight, acceleration, displacement, cylinders, horsepower, model_year, origin)

    # Scale the input data
    scaler = StandardScaler()
    scaled_input = scaler.fit_transform(preprocessed_input)

    # Predict using the loaded model
    prediction = lr_model.predict(scaled_input)
    return prediction[0]

