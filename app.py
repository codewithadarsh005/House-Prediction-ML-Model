from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

MODEL_FILE = "model.pkl"
PIPELINE_FILE = "pipeline.pkl"

model = joblib.load(MODEL_FILE)
pipeline = joblib.load(PIPELINE_FILE)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = {
        "longitude": float(request.form["longitude"]),
        "latitude": float(request.form["latitude"]),
        "housing_median_age": float(request.form["housing_median_age"]),
        "total_rooms": float(request.form["total_rooms"]),
        "total_bedrooms": float(request.form["total_bedrooms"]),
        "population": float(request.form["population"]),
        "households": float(request.form["households"]),
        "median_income": float(request.form["median_income"]),
        "ocean_proximity": request.form["ocean_proximity"]
    }

    df = pd.DataFrame([data])
    transformed = pipeline.transform(df)
    prediction = model.predict(transformed)[0]

    return render_template(
        "index.html",
        prediction=round(prediction, 2)
    )
