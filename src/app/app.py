from flask import Flask, render_template, request, jsonify, redirect, url_for, session
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import os
import sqlite3
from datetime import datetime

app = Flask(__name__)
app.secret_key = "super_secret_key_change_this"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
MODEL_PATH = os.path.join(BASE_DIR, "models", "saved_model")
DB_PATH = os.path.join(BASE_DIR, "data", "processed", "logs.db")

MODEL_NAME = "DistilBERT-base-uncased"
EPOCHS_TRAINED = 2
MACRO_F1 = 0.7446
DATASET_SIZE = 281197
TRAIN_SIZE = int(DATASET_SIZE * 0.9)
VAL_SIZE = DATASET_SIZE - TRAIN_SIZE

labels = [
    "hate",
    "offensive",
    "aggression",
    "bullying",
    "toxic",
    "racism",
    "sexism",
    "religious_hate"
]

# ---------------- MODEL ----------------

tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
model = DistilBertForSequenceClassification.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# ---------------- DATABASE ----------------

def init_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE,
                  password TEXT)''')

    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  user_id INTEGER,
                  text TEXT,
                  risk_level TEXT,
                  timestamp TEXT,
                  FOREIGN KEY(user_id) REFERENCES users(id))''')

    conn.commit()
    conn.close()

init_db()

# ---------------- MODEL PREDICTION ----------------

def predict(text):
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )

    input_ids = encoding["input_ids"].to(DEVICE)
    attention_mask = encoding["attention_mask"].to(DEVICE)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.sigmoid(outputs.logits)

    return dict(zip(labels, probs.cpu().numpy()[0]))

# ---------------- REGISTER ----------------

@app.route("/register", methods=["GET", "POST"])
def register():
    error = None
    success = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()

        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)",
                      (username, password))
            conn.commit()
            success = "Account created successfully."
        except sqlite3.IntegrityError:
            error = "Username already exists."

        conn.close()

    return render_template("register.html", error=error, success=success)

# ---------------- LOGIN ----------------

@app.route("/login", methods=["GET", "POST"])
def login():
    error = None

    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("SELECT id FROM users WHERE username=? AND password=?",
                  (username, password))
        user = c.fetchone()
        conn.close()

        if user:
            session["logged_in"] = True
            session["user_id"] = user[0]
            session["username"] = username
            return redirect(url_for("dashboard"))
        else:
            error = "Invalid credentials"

    return render_template("login.html", error=error)

@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("login"))

# ---------------- MAIN PAGE ----------------

@app.route("/", methods=["GET", "POST"])
def index():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    predictions = None
    risk_level = None

    if request.method == "POST":
        text = request.form["text"]
        predictions = predict(text)

        max_prob = max(predictions.values())

        if max_prob > 0.75:
            risk_level = "High Risk"
        elif max_prob > 0.4:
            risk_level = "Moderate Risk"
        else:
            risk_level = "Safe"

        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        c.execute("INSERT INTO logs (user_id, text, risk_level, timestamp) VALUES (?, ?, ?, ?)",
                  (session["user_id"], text, risk_level,
                   datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        conn.commit()
        conn.close()

    return render_template("index.html",
                           predictions=predictions,
                           risk_level=risk_level,
                           username=session.get("username"))

# ---------------- DASHBOARD WITH HISTORY ----------------

@app.route("/dashboard")
def dashboard():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Stats
    c.execute("SELECT risk_level, COUNT(*) FROM logs WHERE user_id=? GROUP BY risk_level",
              (session["user_id"],))
    stats = dict(c.fetchall())

    # Last 10 activities
    c.execute("SELECT text, risk_level, timestamp FROM logs WHERE user_id=? ORDER BY id DESC LIMIT 10",
              (session["user_id"],))
    history = c.fetchall()

    conn.close()

    return render_template("dashboard.html",
                           stats=stats,
                           history=history,
                           username=session.get("username"))

# ---------------- PROFILE ----------------

@app.route("/profile")
def profile():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT COUNT(*) FROM logs WHERE user_id=?", (session["user_id"],))
    total_analyses = c.fetchone()[0]

    conn.close()

    return render_template("profile.html",
                           username=session.get("username"),
                           total_analyses=total_analyses)

# ---------------- MODEL INFO ----------------

@app.route("/model-info")
def model_info():
    if not session.get("logged_in"):
        return redirect(url_for("login"))

    return render_template("model_info.html",
                           model_name=MODEL_NAME,
                           epochs=EPOCHS_TRAINED,
                           macro_f1=MACRO_F1,
                           dataset_size=DATASET_SIZE,
                           train_size=TRAIN_SIZE,
                           val_size=VAL_SIZE,
                           device=str(DEVICE),
                           labels=labels)

# ---------------- API ----------------

@app.route("/api/predict", methods=["POST"])
def api_predict():
    data = request.json
    text = data.get("text", "")
    predictions = predict(text)
    return jsonify(predictions)

if __name__ == "__main__":
    app.run(debug=True)
