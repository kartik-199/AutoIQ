import joblib

loaded_model = joblib.load('multiple_regression_model_mpg.pkl')

prediction = loaded_model.predict([[180, 6, 120, 250, 10.5, 1970, 1]])
print(prediction)
