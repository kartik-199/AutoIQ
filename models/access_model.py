import joblib
from ucimlrepo import fetch_ucirepo
import math
'''
loaded_model = joblib.load('multiple_regression_model_mpg.pkl')

prediction = loaded_model.predict([[180, 6, 120, 250, 10.5, 1970, 1]])
print(prediction)
'''
auto_mpg = fetch_ucirepo(id=9)
features = auto_mpg.data.features
print(features.columns)

def get_model_prediction(displacement, cylinders, horsepower, acceleration, model_year, origin):
    input_instance = [[displacement, cylinders, horsepower, acceleration, model_year, origin]]
    loaded_model = joblib.load('multiple_regression_model_mpg.pkl')
    prediction = loaded_model.predict(input_instance)
    prediction = f"{prediction[0][0]: .2f}"
    return prediction

