
# 📩 Spam SMS Detector 🔍🤖

An AI-powered web application that detects whether an SMS message is **Spam 🚨** or **Not Spam ✅** using Machine Learning.

Live Demo:
👉 [https://mydailywork.onrender.com](https://mydailywork.onrender.com)

---

## 🚀 Features

✨ Interactive and colorful UI
✨ Real-time Spam Detection
✨ Pre-filled example messages
✨ Emoji-enhanced modern design
✨ Machine Learning based prediction
✨ Deployed on Render

---

## 🧠 Machine Learning Model

The application uses:

* **TF-IDF Vectorizer**
* **LinearSVC Classifier**
* Trained on SMS Spam Dataset
* Scikit-learn based pipeline

---

## 🛠 Tech Stack

* Python 🐍
* Flask 🌐
* Scikit-learn 🤖
* HTML + CSS 🎨
* Gunicorn 🚀
* Render (Deployment)

---

## 📂 Project Structure

```
MyDailyWork_SpamTask/
│
├── app.py
├── train_model.py
├── spam_model.py
├── model.pkl
├── vectorizer.pkl
├── spam.csv
├── requirements.txt
├── templates/
│   └── index.html
└── README.md
```

---

## ⚙ Installation (Run Locally)

### 1️⃣ Clone Repository

```
git clone https://github.com/rbkroyal009/MYDAILYWORK.git
cd MYDAILYWORK
```

### 2️⃣ Create Virtual Environment

```
python -m venv venv
venv\Scripts\activate
```

### 3️⃣ Install Dependencies

```
pip install -r requirements.txt
```

### 4️⃣ Train Model (If Needed)

```
python train_model.py
```

### 5️⃣ Run Application

```
python app.py
```

Open browser:

```
http://127.0.0.1:5000
```

---

## 🌍 Deployment

This project is deployed using:

**Render Web Service**

Deployment command used:

```
gunicorn app:app
```

---

## 📊 How It Works

1️⃣ User enters SMS message
2️⃣ Message is transformed using TF-IDF
3️⃣ Model predicts spam or not spam
4️⃣ Result displayed instantly

---

## 🎯 Example Test Messages

Spam Example:

```
Congratulations! You won a free iPhone! Click now!
```

Normal Message:

```
Hey, are we meeting at 5 pm today?
```

---

## 🔒 Model Notes

* Trained using Scikit-learn 1.7.2
* Compatible version pinned in requirements.txt
* Model stored as `.pkl` files

---

## 💡 Future Improvements

* Confidence score display
* Message history tracking
* Dark mode toggle
* REST API endpoint
* Docker deployment
* Database integration

---

## 👨‍💻 Author

**Bharath Kumar Ramisetti**

Machine Learning & Networking Student
Passionate about AI-powered applications 🚀

---

## ❤️ Acknowledgement

Built as part of Machine Learning Internship project.

---
