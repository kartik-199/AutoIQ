# AutoIQ


This project is a Flask web application that predicts the fuel efficiency (miles per gallon, MPG) of cars based on various features such as weight, acceleration, displacement, and others. The app also tracks and displays a leaderboard of the top predictions made by users.

## Features

- **Car Prediction**: Users can input car features such as weight, acceleration, displacement, cylinders, horsepower, model year, and origin to predict the car's MPG.
- **Leaderboard**: Displays the top 10 predictions based on MPG, sorted in descending order.
- **Data Storage**: Leaderboard data is stored in a `leaderboard.json` file and automatically updated with each new prediction.
- **Model**: The model is a multiple linear regression model trained on a dataset of automobile MPG values, including engineered features.

## Tech Stack

- **Backend**: Flask (Python web framework)
- **Machine Learning**: Scikit-learn (for model training and preprocessing)
- **Data Storage**: JSON (for leaderboard storage)
- **Frontend**: HTML (with basic templates for displaying results)
- **Libraries**: Pandas, Numpy, Joblib, Scikit-learn, Flask

## Installation

### Prerequisites

Before running the app, make sure you have the following installed:
- Python 3.7 or higher
- Flask
- Scikit-learn
- Pandas
- Numpy
- Joblib

You can install the required dependencies using pip: pip install -r requirements.txt


### Running the App

1. Clone the repository or download the project files.
2. Make sure you have trained the model (`multiple_regression_model_mpg_engineered.pkl`) using the provided model training script.
3. Run the Flask application: python app.py

4. Visit `http://localhost:5000` in your web browser to use the application.

## Files Overview

- **app.py**: The main Flask application file. It contains routes for displaying the homepage, the prediction page, and the leaderboard page.
- **leaderboard.json**: Stores the leaderboard data, which includes the top predictions and their corresponding car features.
- **templates/**: Folder containing HTML files for rendering the web pages.
  - `index.html`: The home page.
  - `predict.html`: The prediction form where users can input car features.
  - `leaderboard.html`: Displays the leaderboard with the top predictions.
- **models/**: Folder containing the trained model and the model preprocessing scripts.
  - `multiple_regression_model_mpg_engineered.pkl`: The trained machine learning model file.
  - `access_model.py`: Contains functions for preprocessing inputs and getting predictions from the model.

## How it Works

### 1. **User Input**:
   - Users input car features such as weight, acceleration, displacement, cylinders, horsepower, model year, and origin through a web form.

### 2. **Prediction**:
   - The app preprocesses the inputs using feature engineering techniques such as interaction terms, transformations, and polynomial features.
   - The preprocessed input is then passed to the machine learning model to predict the MPG of the car.

### 3. **Leaderboard**:
   - Each prediction is added to a leaderboard stored in a `leaderboard.json` file. The leaderboard is sorted by MPG, and only the top 10 predictions are retained.

### 4. **Model Training**:
   - The model used for predictions is a multiple regression model trained on a dataset containing various car features. It includes engineered features for better predictive performance.

## Example of Preprocessing

The input features undergo preprocessing, including:
- Creating interaction terms between weight and acceleration, and cylinders and displacement.
- Applying power transformations like the logarithm of weight and square root of displacement.
- Encoding the origin of the car as binary categorical variables.
- Binning the horsepower variable into discrete categories.
- Generating polynomial features based on weight, acceleration, and displacement.

## Contributing

Feel free to fork the repository and contribute improvements, bug fixes, or new features!

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.



