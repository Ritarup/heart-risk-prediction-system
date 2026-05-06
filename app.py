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


model = None

def load_model():
    global model
    if model is None:
        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        model = joblib.load(model_path)

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
        if users.find_one({"username": username}):
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))
        
        users.insert_one({"username": username, "password": generate_password_hash(password)})
        flash("Registration successful. Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        user = users.find_one({"username": request.form['username']})
        if user and check_password_hash(user["password"], request.form['password']):
            session['user'] = user["username"]
            return redirect(url_for('home'))
        flash("Invalid username or password", "danger")
    return render_template('login.html')

@app.route('/predict', methods=['POST'])
def predict():
    load_model() 
    try:
        feature_names = ['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo', 'cholesterol', 'gluc', 'smoke', 'alco', 'active']
        defaults = {'age': 53, 'gender': 1, 'height': 165, 'weight': 74, 'ap_hi': 120, 'ap_lo': 80, 'cholesterol': 1, 'gluc': 1, 'smoke': 0, 'alco': 0, 'active': 1}

        final_input = []
        missing_count = 0
        for f in feature_names:
            val = request.form.get(f)
            if val and val.strip():
                final_input.append(float(val))
            else:
                final_input.append(float(defaults[f]))
                missing_count += 1

        prediction = model.predict(np.array(final_input).reshape(1, -1))[0]
        result_text = "High Risk" if prediction == 1 else "Low Risk"

       
        reasons = []
        if final_input[4] >= 140: reasons.append("High systolic BP")
        if final_input[5] >= 90: reasons.append("High diastolic BP")
        if final_input[6] > 1: reasons.append("High cholesterol")
        if final_input[0] > 55: reasons.append("Age over 55")
        
        factor_msg = ", ".join(reasons) if reasons else "General health indicators."
        warning_text = f"{missing_count} fields were missing — population averages used." if missing_count > 0 else ""

        return render_template('index.html', prediction_text=result_text, factor=factor_msg, warning=warning_text)
    except Exception as e:
        return render_template('index.html', warning=f"Error: {str(e)}")

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('login'))

if __name__ == "__main__":
    load_model()
    app.run()