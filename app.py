"""Flask backend: serves the page, predicts, and stores every check in SQLite."""
import csv
import io
import json
import sqlite3
import sys
from datetime import datetime

import joblib
import pandas as pd
from flask import Flask, Response, jsonify, render_template, request

app = Flask(__name__)
try:
    model = joblib.load("model.joblib")
except FileNotFoundError:
    sys.exit("model.joblib not found. Run 'python train_model.py' first, then start the app again.")
DB = "loan.db"
LABELS = {"ApplicantIncome": "Applicant income", "CoapplicantIncome": "Co-applicant income", "LoanAmount": "Loan amount",
          "Loan_Amount_Term": "Loan term", "Credit_History": "Credit history", "Married": "Marital status",
          "Dependents": "Dependents", "Education": "Education", "Self_Employed": "Self-employment",
          "Property_Area": "Property area"}
try:  # a "typical applicant", saved by train_model.py
    with open("baseline.json") as f:
        BASELINE = json.load(f)
except FileNotFoundError:
    BASELINE = None
ANNUAL_RATE = 9.0  # assumed interest rate (%) used only for the EMI estimate

NUMBERS = {"ApplicantIncome": (0, 1_000_000), "CoapplicantIncome": (0, 1_000_000),
           "LoanAmount": (1, 10_000), "Loan_Amount_Term": (12, 600)}
CHOICES = {"Married": ["Yes", "No"], "Dependents": ["0", "1", "2", "3+"],
           "Education": ["Graduate", "Not Graduate"], "Self_Employed": ["Yes", "No"],
           "Credit_History": ["1", "0"], "Property_Area": ["Urban", "Semiurban", "Rural"]}
FEATURES = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term",
            "Credit_History", "Married", "Dependents", "Education", "Self_Employed", "Property_Area"]


def db():
    conn = sqlite3.connect(DB)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    with db() as c:
        c.execute("""CREATE TABLE IF NOT EXISTS applications (
            id INTEGER PRIMARY KEY AUTOINCREMENT, created_at TEXT,
            ApplicantIncome REAL, CoapplicantIncome REAL, LoanAmount REAL, Loan_Amount_Term REAL,
            Credit_History INTEGER, Married TEXT, Dependents TEXT, Education TEXT,
            Self_Employed TEXT, Property_Area TEXT, probability REAL, decision TEXT)""")


def validate(data):
    clean, errors = {}, {}
    for key, (low, high) in NUMBERS.items():
        try:
            value = float(data.get(key))
            if not low <= value <= high:
                raise ValueError
            clean[key] = value
        except (TypeError, ValueError):
            errors[key] = f"{key} must be a number between {low} and {high}."
    for key, allowed in CHOICES.items():
        if str(data.get(key)) in allowed:
            clean[key] = str(data[key])
        else:
            errors[key] = f"{key} must be one of: {', '.join(allowed)}."
    if not errors:
        clean["Credit_History"] = int(clean["Credit_History"])
    return clean, errors


def drivers(clean, p_full):
    """Which inputs moved this result? Swap each input for the typical value and see how much the score changes."""
    if not BASELINE:
        return []
    out = []
    for key in FEATURES:
        p = float(model.predict_proba(pd.DataFrame([{**clean, key: BASELINE[key]}])[FEATURES])[0][1])
        out.append({"label": LABELS[key], "effect": round((p_full - p) * 100)})
    out = [d for d in out if abs(d["effect"]) >= 1]
    return sorted(out, key=lambda d: -abs(d["effect"]))[:5]


def emi(amount, months):
    r = ANNUAL_RATE / 1200
    return amount * r * (1 + r) ** months / ((1 + r) ** months - 1)


@app.get("/")
def home():
    return render_template("index.html")


@app.post("/api/predict")
def predict():
    clean, errors = validate(request.get_json(silent=True) or {})
    if errors:
        return jsonify(errors=errors), 400

    raw = float(model.predict_proba(pd.DataFrame([clean])[FEATURES])[0][1])
    approved = raw >= 0.5
    probability = min(max(raw, 0.01), 0.99)  # never show a flat 0% or 100%

    monthly_emi = emi(clean["LoanAmount"] * 1000, clean["Loan_Amount_Term"])
    income = clean["ApplicantIncome"] + clean["CoapplicantIncome"]
    share = monthly_emi / income * 100 if income else 100

    reasons = []
    reasons.append("A good credit history is the strongest point in your favour." if clean["Credit_History"]
                   else "No credit history or past defaults is the biggest risk factor.")
    reasons.append(f"The EMI would take {share:.0f}% of household income, which is comfortable." if share <= 35
                   else f"The EMI would take {share:.0f}% of household income. Above 35% worries lenders.")
    suggestion = None
    if not approved:  # what-if: lower the loan in 5% steps until the model says yes
        for step in range(1, 20):
            amount = round(clean["LoanAmount"] * (1 - 0.05 * step) / 10) * 10
            if amount < 10:
                break
            trial = pd.DataFrame([{**clean, "LoanAmount": amount}])[FEATURES]
            if model.predict_proba(trial)[0][1] >= 0.5:
                suggestion = f"Reducing the loan to about ₹{int(amount * 1000):,} should make approval likely."
                break
        reasons.append(suggestion or "A smaller loan or a longer term lowers the EMI and can improve the result.")

    warnings = []
    if income > 50000 or clean["LoanAmount"] > 500:
        warnings.append("These figures are larger than most of the training data, so treat this estimate as rough.")

    with db() as c:
        c.execute("""INSERT INTO applications (created_at, ApplicantIncome, CoapplicantIncome, LoanAmount,
                     Loan_Amount_Term, Credit_History, Married, Dependents, Education, Self_Employed,
                     Property_Area, probability, decision)
                     VALUES (:t, :ApplicantIncome, :CoapplicantIncome, :LoanAmount, :Loan_Amount_Term,
                     :Credit_History, :Married, :Dependents, :Education, :Self_Employed,
                     :Property_Area, :p, :d)""",
                  {**clean, "t": datetime.now().strftime("%d %b %H:%M"), "p": probability,
                   "d": "Approved" if approved else "Declined"})

    band = "likely" if probability >= 0.70 else "borderline" if probability >= 0.45 else "unlikely"
    return jsonify(approved=approved, probability=probability, band=band, emi=monthly_emi, emi_share=share,
                   reasons=reasons, warnings=warnings, drivers=drivers(clean, raw))


@app.get("/api/history")
def history():
    with db() as c:
        rows = c.execute("SELECT * FROM applications ORDER BY id DESC LIMIT 8").fetchall()
    return jsonify([dict(r) for r in rows])


@app.get("/insights")
def insights_page():
    return render_template("insights.html")


@app.get("/api/insights")
def insights():
    try:
        with open("metrics.json") as f:
            metrics = json.load(f)
    except FileNotFoundError:
        metrics = None  # user has not run train_model.py yet
    with db() as c:
        usage = c.execute("SELECT decision, COUNT(*) AS n FROM applications GROUP BY decision").fetchall()
    return jsonify(metrics=metrics, usage=[dict(r) for r in usage])


@app.get("/api/export.csv")
def export_csv():
    with db() as c:
        cur = c.execute("SELECT * FROM applications ORDER BY id")
        columns = [d[0] for d in cur.description]
        rows = cur.fetchall()
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(columns)
    writer.writerows([tuple(r) for r in rows])
    return Response(buffer.getvalue(), mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=loanlens_history.csv"})


@app.delete("/api/history")
def clear_history():
    with db() as c:
        c.execute("DELETE FROM applications")
    return jsonify(ok=True)


init_db()
if __name__ == "__main__":
    app.run(debug=True)
