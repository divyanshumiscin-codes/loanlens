"""Creates a practice dataset (data/loan_data.csv) with the same columns as the
Kaggle 'Loan Prediction' dataset. It is only a fallback for when the real
dataset is missing, and it never overwrites an existing file."""
import os
import sys

import numpy as np
import pandas as pd

OUTPUT = "data/loan_data.csv"
if os.path.exists(OUTPUT):  # protect the real dataset
    print(OUTPUT, "already exists, so nothing was changed.")
    sys.exit()
os.makedirs("data", exist_ok=True)

rng = np.random.default_rng(42)
n = 1000
df = pd.DataFrame({
    "Married": rng.choice(["Yes", "No"], n, p=[.65, .35]),
    "Dependents": rng.choice(["0", "1", "2", "3+"], n, p=[.55, .17, .17, .11]),
    "Education": rng.choice(["Graduate", "Not Graduate"], n, p=[.78, .22]),
    "Self_Employed": rng.choice(["Yes", "No"], n, p=[.14, .86]),
    "ApplicantIncome": rng.lognormal(8.5, .5, n).round(),
    "CoapplicantIncome": np.where(rng.random(n) < .45, 0, rng.lognormal(7.6, .6, n)).round(),
    "LoanAmount": rng.normal(140, 60, n).clip(20, 500).round(),
    "Loan_Amount_Term": rng.choice([120, 180, 240, 300, 360, 480], n, p=[.02, .05, .03, .02, .82, .06]),
    "Credit_History": rng.choice([1, 0], n, p=[.84, .16]),
    "Property_Area": rng.choice(["Urban", "Semiurban", "Rural"], n),
})
# Hidden "bank rule" the model will try to learn: credit history and a
# small EMI-to-income ratio matter most.
ratio = (df.LoanAmount * 1000 / df.Loan_Amount_Term) / (df.ApplicantIncome + df.CoapplicantIncome)
score = (-1 + 3.5 * df.Credit_History - 8 * ratio
         + 0.4 * (df.Education == "Graduate") + 0.3 * (df.Property_Area == "Semiurban")
         + rng.normal(0, .8, n))
df["Loan_Status"] = np.where(score > 0.5, "Y", "N")
df.insert(0, "Loan_ID", [f"LP{1000 + i}" for i in range(n)])
df.to_csv(OUTPUT, index=False)
print("Saved", OUTPUT, df.shape, "| approval rate:", round((df.Loan_Status == "Y").mean(), 2))