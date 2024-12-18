from models.access_model import get_model_prediction
import json
from flask import Flask, render_template, request
from datetime import datetime

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods = ['POST'])
def predict():
    # Get prediction
    displacement = float(request.form['displacement'])
    cylinders = float(request.form['cylinders'])
    horsepower = float(request.form['cylinders'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['weight'])
    model_year = int(request.form['model_year'])
    origin = int(request.form['origin'])
    prediction = get_model_prediction(weight, acceleration, displacement, cylinders, horsepower, model_year, origin)
    prediction = f"{prediction: .2f}"

    # Acces and modify leaderboard data if necessary
    new_entry = {
        "displacement": displacement,
        "cylinders": cylinders,
        "mpg": prediction,
        "date": datetime.now().strftime("%m-%d-%y %H:%M")
    }
    try:
        with open("leaderboard.json", "r") as file:
            leaderboard_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        leaderboard_data = []
    leaderboard_data.append(new_entry)
    leaderboard_data = sorted(leaderboard_data, key=lambda x: x["mpg"], reverse=True)  # Sort by MPG (descending)
    leaderboard_data = leaderboard_data[:10]
    with open("leaderboard.json", "w") as file:
        json.dump(leaderboard_data, file, indent=4)
    # Return predict template
    return render_template(
        'predict.html',
        prediction=prediction,
        displacement=displacement,
        cylinders=cylinders,
        horsepower=horsepower,
        weight=weight,
        acceleration=acceleration,
        model_year=model_year,
        origin=origin
    )


@app.route('/leaderboard')
def leaderboard_page():
    try:
        with open("leaderboard.json", "r") as file:
            leaderboard_data = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        leaderboard_data = []
    return render_template('leaderboard.html', leaderboard=leaderboard_data)

if __name__ == '__main__':
    app.run(debug=True)