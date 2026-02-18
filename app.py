from flask import Flask, render_template, request
import pickle
import re
import string

app = Flask(__name__)

# Load saved model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

@app.route("/", methods=["GET", "POST"])
def home():
    prediction = ""

    if request.method == "POST":
        message = request.form["message"]
        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)

        prediction = "Spam ❌" if result[0] == 1 else "Not Spam ✅"

    return render_template("index.html", prediction=prediction)
