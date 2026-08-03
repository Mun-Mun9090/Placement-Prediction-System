import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template
from src.data.load_data import load_data, get_summary

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/dataset")
def dataset():
    df=load_data()
    summary=get_summary(df)
    return render_template("dataset.html", summary=summary, first_rows=df.head().to_html(index=False))

@app.route("/eda")
def eda():
    return render_template("eda.html")

@app.route("/preprocessing")
def preprocessing():
    return render_template("preprocessing.html")

@app.route("/models")
def models():
    return render_template("models.html")

@app.route("/evaluation")
def evaluation():
    return render_template("evaluation.html")

@app.route("/comparison")
def comparison():
    return render_template("comparison.html")

@app.route("/prediction")
def prediction():
    return render_template("prediction.html")

if __name__ == "__main__":
    app.run(debug=True)
