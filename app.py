from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load trained model and vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    message = request.form.get("message")

    if not message:
        return render_template("index.html", prediction="⚠️ Please enter a message")

    data = vectorizer.transform([message])
    prediction = model.predict(data)[0]

    if prediction == 1:
        result = "🚨 SPAM MESSAGE"
    else:
        result = "✅ NOT SPAM"

    return render_template("index.html", prediction=result, user_message=message)


# Important for Render
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port)
