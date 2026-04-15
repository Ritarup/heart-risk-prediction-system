from flask import Flask, render_template, request, redirect, url_for, session, flash
import joblib
import numpy as np
import sqlite3
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
app.secret_key = 'heart_ai_secure_key'

model = joblib.load('model.pkl')


def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY,
            username TEXT UNIQUE,
            password TEXT
        )
    ''')
    conn.commit()
    conn.close()

init_db()


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

        
        hashed_pw = generate_password_hash(password)

        conn = sqlite3.connect('database.db')
        c = conn.cursor()

   
        existing_user = c.execute(
            "SELECT * FROM users WHERE username = ?",
            (username,)
        ).fetchone()

        if existing_user:
            conn.close()
            flash("Username already exists!", "danger")
            return redirect(url_for('register'))

        c.execute(
            "INSERT INTO users (username, password) VALUES (?, ?)",
            (username, hashed_pw)
        )
        conn.commit()
        conn.close()

        flash("Registration successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        conn = sqlite3.connect('database.db')
        user = conn.execute(
            "SELECT * FROM users WHERE username = ?",
            (request.form['username'],)
        ).fetchone()
        conn.close()

        if user and check_password_hash(user[2], request.form['password']):
            session['user'] = user[1]
            return redirect(url_for('home'))
        else:
            flash('Invalid username or password', 'danger')

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
        prob = model.predict_proba(final_input)[0][1]

        result_text = "High Risk" if prob > 0.6 else "Low Risk"

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
            prob=round(prob * 100, 2),
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
