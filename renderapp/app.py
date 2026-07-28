from flask import Flask, render_template, request
from model import recommend

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():

    recommendations = []

    if request.method == "POST":
        movie = request.form["movie"]
        recommendations = recommend(movie)

    return render_template(
        "index.html",
        recommendations=recommendations
    )

if __name__ == "__main__":
    app.run(debug=True)