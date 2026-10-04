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

# CIBIL layer: the dataset has no CIBIL scores, so a score is turned into the dataset's credit-history signal.
CIBIL_WEAK, CIBIL_GOOD = 650, 750  # below 650 = weak, 750 and above = good, in between = borderline
CIBIL_TEXT = {
    "weak": "Your CIBIL score of {score} is below 650, which lenders generally see as weak. The model treats it as a poor credit history.",
    "borderline": "Your CIBIL score of {score} is in the borderline zone (650 to 749). The model averages the poor-credit and good-credit results.",
    "good": "Your CIBIL score of {score} is in the good zone (750 and above). The model treats it as a good credit history.",
}


def cibil_zone(score):
    return "good" if score >= CIBIL_GOOD else "borderline" if score >= CIBIL_WEAK else "weak"


def shown(p):
    return min(max(p, 0.01), 0.99)  # never show a flat 0% or 100%


def chance(inputs, cibil=None):
    """Chance of approval from the model. A CIBIL score replaces the credit-history input:
    good = good credit, weak = poor credit, borderline = the average of the two."""
    def run(credit=None):
        row = dict(inputs) if credit is None else {**inputs, "Credit_History": credit}
        return float(model.predict_proba(pd.DataFrame([row])[FEATURES])[0][1])
    if cibil is None:
        return run()
    zone = cibil_zone(cibil)
    if zone == "good":
        return run(1)
    if zone == "weak":
        return run(0)
    return (run(0) + run(1)) / 2


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
    if data.get("Cibil_Score") not in (None, ""):  # optional
        try:
            score = float(data["Cibil_Score"])
            if not 300 <= score <= 900:
                raise ValueError
            clean["Cibil_Score"] = score
        except (TypeError, ValueError):
            errors["Cibil_Score"] = "Cibil_Score must be a number between 300 and 900."
    if not errors:
        clean["Credit_History"] = int(clean["Credit_History"])
    return clean, errors


def drivers(clean, p_full, cibil=None):
    """Which inputs moved this result? Swap each input for the typical value and see how much the score changes."""
    if not BASELINE:
        return []
    out = []
    for key in FEATURES:
        if key == "Credit_History" and cibil is not None:
            continue  # the CIBIL score replaces this input
        p = chance({**clean, key: BASELINE[key]}, cibil)
        out.append({"label": LABELS[key], "effect": round((p_full - p) * 100)})
    if cibil is not None:  # compare the CIBIL score with a typical applicant, who has good credit
        p = chance({**clean, "Credit_History": BASELINE["Credit_History"]})
        out.append({"label": "CIBIL score", "effect": round((p_full - p) * 100)})
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
    cibil = clean.pop("Cibil_Score", None)  # optional: None means "not given"

    raw = chance(clean, cibil)
    approved = raw >= 0.5
    probability = shown(raw)
    at_weak, at_mid, at_good = chance(clean, 300), chance(clean, 700), chance(clean, 900)

    monthly_emi = emi(clean["LoanAmount"] * 1000, clean["Loan_Amount_Term"])
    income = clean["ApplicantIncome"] + clean["CoapplicantIncome"]
    share = monthly_emi / income * 100 if income else 100

    reasons = []
    if cibil is not None:
        reasons.append(CIBIL_TEXT[cibil_zone(cibil)].format(score=int(cibil)))
    else:
        reasons.append("A good credit history is the strongest point in your favour." if clean["Credit_History"]
                       else "No credit history or past defaults is the biggest risk factor.")
    reasons.append(f"The EMI would take {share:.0f}% of household income, which is comfortable." if share <= 35
                   else f"The EMI would take {share:.0f}% of household income. Above 35% worries lenders.")
    reasons.append(f"CIBIL effect: with a score of 750 or more the chance would be about {shown(at_good):.0%}, "
                   f"and with a score below 650 about {shown(at_weak):.0%}.")
    suggestion = None
    if not approved:  # what-if: lower the loan in 5% steps until the model says yes
        for step in range(1, 20):
            amount = round(clean["LoanAmount"] * (1 - 0.05 * step) / 10) * 10
            if amount < 10:
                break
            if chance({**clean, "LoanAmount": amount}, cibil) >= 0.5:
                suggestion = f"Reducing the loan to about ₹{int(amount * 1000):,} should make approval likely."
                break
        reasons.append(suggestion or "A smaller loan or a longer term lowers the EMI and can improve the result.")

    warnings = []
    if income > 50000 or clean["LoanAmount"] > 500:
        warnings.append("These figures are larger than most of the training data, so treat this estimate as rough.")

    if cibil is not None:  # store 1 when the score is 650 or more
        clean["Credit_History"] = 1 if cibil >= CIBIL_WEAK else 0
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
    meter = dict(used=cibil is not None, score=cibil, zone=cibil_zone(cibil) if cibil is not None else None,
                 chances=dict(weak=shown(at_weak), borderline=shown(at_mid), good=shown(at_good)))
    return jsonify(approved=approved, probability=probability, band=band, emi=monthly_emi, emi_share=share,
                   reasons=reasons, warnings=warnings, drivers=drivers(clean, raw, cibil), cibil=meter)


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