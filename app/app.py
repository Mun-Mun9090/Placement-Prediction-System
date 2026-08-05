import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from flask import Flask, render_template, send_from_directory
from src.data.load_data import load_data, get_summary

app = Flask(__name__, static_folder='static', static_url_path='/static')
# Also serve charts folder as static
charts_path = os.path.join(os.path.dirname(__file__), 'charts')
app.config['CHARTS_FOLDER'] = charts_path

@app.route("/")
def home():
    return render_template("home.html")

@app.route('/charts/<filename>')
def serve_chart(filename):
    charts_path = os.path.join(os.path.dirname(__file__), 'charts')
    return send_from_directory(charts_path, filename)

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
