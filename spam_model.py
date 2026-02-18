from flask import Flask, render_template, request
import pandas as pd
import string
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

app = Flask(__name__)

# ----------------------------------
# 1️⃣ Load Dataset (Excel Version)
# ----------------------------------
df = pd.read_excel("spam.xlsx")

# Keep required columns
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Convert labels to numeric
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# ----------------------------------
# 2️⃣ Clean Text
# ----------------------------------
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

df['message'] = df['message'].apply(clean_text)

# ----------------------------------
# 3️⃣ Train Model
# ----------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

model = LinearSVC()
model.fit(X_train_tfidf, y_train)

# Calculate accuracy
accuracy = accuracy_score(y_test, model.predict(X_test_tfidf))
print("Model Accuracy:", round(accuracy, 4))

# ----------------------------------
# 4️⃣ Web Route
# ----------------------------------
@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = ""

    if request.method == 'POST':
        message = request.form['message']
        cleaned = clean_text(message)
        vectorized = vectorizer.transform([cleaned])
        result = model.predict(vectorized)

        if result[0] == 1:
            prediction = "Spam ❌"
        else:
            prediction = "Not Spam ✅"

    return render_template('index.html', prediction=prediction, accuracy=round(accuracy, 4))

if __name__ == "__main__":
    app.run(debug=True)
