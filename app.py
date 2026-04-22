from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
from dotenv import load_dotenv

import os
from werkzeug.security import generate_password_hash, check_password_hash
from pymongo import MongoClient


load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "heart_ai_secure_key")

MONGO_URI = os.getenv("MONGO_URI")
client = MongoClient(MONGO_URI)
db = client["mydb"]
users = db["users"]

model = joblib.load("model.pkl")




@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        existing_user = users.find_one({"username": username})

        if existing_user:
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        hashed_pw = generate_password_hash(password)

        users.insert_one({
            "username": username,
            "password": hashed_pw
        })

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')
   


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = users.find_one({
            "username": request.form['username']
        })

        if user and check_password_hash(user["password"], request.form['password']):
            session['user'] = user["username"]
            return redirect(url_for('home'))
        else:
            flash("Invalid username or password", "danger")

    return render_template('login.html')
    


@app.route('/predict', methods=['POST'])
def predict():
    try:
        feature_names = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
            'restecg', 'thalach', 'exang', 'oldpeak',
            'slope', 'ca', 'thal'
        ]

        defaults = {
            'age': 54,
            'sex': 1,
            'cp': 1,
            'trestbps': 130,
            'chol': 245,
            'fbs': 0,
            'restecg': 1,
            'thalach': 150,
            'exang': 0,
            'oldpeak': 1.0,
            'slope': 1,
            'ca': 0,
            'thal': 2
        }

        final_input = []
        missing_count = 0

        for f in feature_names:
            val = request.form.get(f)
            if val and val.strip() != "":
                final_input.append(float(val))
            else:
                final_input.append(defaults[f])
                missing_count += 1

        raw = final_input.copy()

        final_input = np.array(final_input).reshape(1, -1)

        prediction = model.predict(final_input)[0]
        

        result_text = "High Risk" if prediction == 1 else "Low Risk"

        reasons = []
        if raw[4] > 240:
            reasons.append("High cholesterol")
        if raw[3] > 140:
            reasons.append("High BP")
        if raw[0] > 60:
            reasons.append("Age factor")
        if raw[7] < 100:
            reasons.append("Low heart rate")
        if raw[2] > 0:
            reasons.append("Chest pain")

        factor_msg = ", ".join(reasons) if reasons else "Based on overall health indicators."

        warning_text = ""
        if missing_count > 0:
            warning_text = f"{missing_count} field(s) were missing — clinical averages used."

        return render_template(
            'index.html',
            prediction_text=result_text,
            factor=factor_msg,
            warning=warning_text
        )

    except Exception as e:
        return render_template('index.html', warning=f"Error: {str(e)}")


@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))


if __name__ == "__main__":
    app.run()
