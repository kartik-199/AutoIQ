from models.access_model import get_model_prediction

from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict')
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods = ['POST'])
def predict():
    displacement = float(request.form['displacement'])
    cylinders = float(request.form['cylinders'])
    horsepower = float(request.form['cylinders'])
    weight = float(request.form['weight'])
    acceleration = float(request.form['weight'])
    model_year = int(request.form['model_year'])
    origin = int(request.form['origin'])
    prediction = get_model_prediction(weight, acceleration, displacement, cylinders, horsepower, model_year, origin)
    prediction = f"{prediction: .2f}"
    return render_template('predict.html', prediction=prediction, displacement=displacement,
                           cylinders=cylinders, horsepower=horsepower, weight=weight, acceleration=acceleration,
                           model_year=model_year, origin=origin)


@app.route('/leaderboard')
def leaderboard_page():
    return render_template('leaderboard.html')

if __name__ == '__main__':
    app.run(debug=True)