from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
# Load the pre-trained model
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')
def home():
    return render_template("index.html")
@app.route("/predict", methods=["POST"])
def predict():
    # Get feature inputs from the form
    sepal_length = float(request.form.get("sepal_length"))
    sepal_width = float(request.form.get("sepal_width"))
    petal_length = float(request.form.get("petal_length"))
    petal_width = float(request.form.get("petal_width"))

    # Make a prediction
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    prediction = model.predict(features)[0]
    # Map the numerical prediction to the corresponding species
    species_mapping = {0: "setosa", 1: "versicolor", 2: "virginica"}
    predicted_species = species_mapping[prediction]

    return render_template("result.html", predicted_species=predicted_species)

if __name__ == "__main__":
    app.run(port=5555,debug=True)
