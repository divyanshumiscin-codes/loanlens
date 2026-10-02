# LoanLens: Loan Approval Predictor

Flask backend + scikit-learn model + SQLite database + plain HTML/CSS/JS frontend.

## Run it (Windows, VS Code terminal)
```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python train_model.py
python app.py
```
Open http://127.0.0.1:5000

## Files
- `generate_data.py` makes practice data (replace data/loan_data.csv with the real Kaggle file for submission)
- `train_model.py` trains the model, prints accuracy, saves model.joblib
- `app.py` the server: validates input, predicts, saves to loan.db, returns JSON
- `templates/index.html`, `static/style.css`, `static/app.js` the web page
