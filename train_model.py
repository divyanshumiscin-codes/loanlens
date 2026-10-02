"""Trains 3 models, compares them fairly, keeps the best one, and saves
metrics.json (used by the Model insights page)."""
import json
import os
import subprocess
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

if not os.path.exists("data/loan_data.csv"):
    subprocess.run([sys.executable, "generate_data.py"], check=True)

df = pd.read_csv("data/loan_data.csv")
NUMERIC = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "Credit_History"]
CATEGORY = ["Married", "Dependents", "Education", "Self_Employed", "Property_Area"]

X = df[NUMERIC + CATEGORY]
y = (df["Loan_Status"] == "Y").astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def make_pipeline(classifier):
    """Step 1: fill blanks, scale numbers, turn words into numbers. Step 2: classify."""
    prep = ColumnTransformer([
        ("num", Pipeline([("fill", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), NUMERIC),
        ("cat", Pipeline([("fill", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore"))]), CATEGORY),
    ])
    return Pipeline([("prep", prep), ("clf", classifier)])


# class_weight="balanced" makes the model pay extra attention to the rarer class (rejections),
# so it misses fewer risky loans.
candidates = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "Decision Tree": DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=6, class_weight="balanced", random_state=42),
}

results, fitted = [], {}
for name, clf in candidates.items():
    pipe = make_pipeline(clf)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="accuracy").mean()  # 5 fair practice exams
    pipe.fit(X_train, y_train)
    test = accuracy_score(y_test, pipe.predict(X_test))                              # the final exam
    results.append({"name": name, "cv": round(cv, 4), "test": round(test, 4)})
    fitted[name] = pipe
    print(f"{name:20s} cross-validation {cv:.3f} | test {test:.3f}")

best_name = max(results, key=lambda r: r["cv"])["name"]  # choose using practice exams, not the final exam
best = fitted[best_name]
pred = best.predict(X_test)
print("\nBest model:", best_name)

FRIENDLY = {"ApplicantIncome": "Applicant income", "CoapplicantIncome": "Co-applicant income", "LoanAmount": "Loan amount",
            "Loan_Amount_Term": "Loan term", "Credit_History": "Credit history"}


def pretty(name):
    """Turn column names like Property_Area_Rural into 'Property area: Rural'."""
    for col in CATEGORY:
        if name.startswith(col + "_"):
            return col.replace("_", " ").capitalize() + ": " + name[len(col) + 1:]
    return FRIENDLY.get(name, name)


# Which inputs mattered most?
names = [n.split("__", 1)[1] for n in best.named_steps["prep"].get_feature_names_out()]
clf = best.named_steps["clf"]
raw = np.abs(clf.coef_[0]) if hasattr(clf, "coef_") else clf.feature_importances_
share = raw / raw.sum() * 100
order = np.argsort(share)[::-1][:8]
importance = [{"feature": pretty(names[i]), "value": round(float(share[i]), 1)} for i in order]

# A "typical applicant" (median / most common value), used to explain individual decisions
baseline = {c: float(X[c].median()) for c in NUMERIC}
baseline.update({c: str(X[c].mode()[0]) for c in CATEGORY})


def rate_table(col, names=None):
    """Approval rate for each value of a column (blank values become 'Unknown')."""
    values = df[col].fillna("Unknown").astype(str)
    rows = []
    for value, group in df.groupby(values):
        rows.append({"label": (names or {}).get(value, value), "n": int(len(group)),
                     "rate": round(float((group["Loan_Status"] == "Y").mean()), 3)})
    return rows


income_labels = ["Under ₹2,500", "₹2,500 to 4,000", "₹4,000 to 6,000", "₹6,000 to 10,000", "Over ₹10,000"]
income_counts = pd.cut(df["ApplicantIncome"], [0, 2500, 4000, 6000, 10000, np.inf], labels=income_labels).value_counts(sort=False)
eda = {
    "rows": int(len(df)), "approval_rate": round(float(y.mean()), 3),
    "tables": {"Credit history": rate_table("Credit_History", {"1.0": "Good credit", "0.0": "Poor credit", "Unknown": "Unknown"}),
               "Property area": rate_table("Property_Area"), "Education": rate_table("Education")},
    "income": [{"label": k, "n": int(v)} for k, v in income_counts.items()],
}

metrics = {
    "best": best_name, "rows": int(len(df)), "test_rows": int(len(y_test)), "models": results,
    "accuracy": round(accuracy_score(y_test, pred), 4),
    "precision": round(precision_score(y_test, pred), 4),
    "recall": round(recall_score(y_test, pred), 4),
    "f1": round(f1_score(y_test, pred), 4),
    "confusion": confusion_matrix(y_test, pred).tolist(),  # [[rejected right, rejected wrong], [approved wrong, approved right]]
    "importance": importance, "eda": eda,
}
joblib.dump(best, "model.joblib")
with open("baseline.json", "w") as f:
    json.dump(baseline, f)
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
print("Accuracy:", metrics["accuracy"], "| Saved model.joblib and metrics.json")
