# ChurnRadar 🎯
### Customer Churn Prediction & Retention Analytics

ChurnRadar is a machine-learning-powered dashboard designed to
analyse customer churn, explore recorded reasons why customers
left, and help businesses plan customer-retention activities.

## Features

- Executive overview and customer analytics
- Machine learning churn prediction
- Customer risk prioritisation
- Recorded churn reason analysis
- Customer-level churn reason reports
- Business insights from historical churn records
- Suggested retention actions
- Retention campaign cost and ROI simulator
- Model performance evaluation
- CSV report downloads
- Interactive Streamlit and Plotly dashboard

## Technology Stack

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Git and GitHub

## Dataset

This project uses a public sample telecommunications churn
dataset from Hugging Face:

https://huggingface.co/datasets/aai510-group1/telco-customer-churn

The combined local dataset currently contains 7,043 rows and
52 columns, according to the local data preparation output.

The dataset represents a fictional telecommunications company.
It is intended for educational analysis and is not live company data.

The raw CSV files are excluded from this repository until their
redistribution terms have been confirmed. The recorded-reasons
page requires the prepared file:

`data/churn_reasons_dataset.csv`

## Run Locally on Windows

1. Install Python 3.10 or a compatible version.
2. Open this project folder in Visual Studio Code.
3. Open Terminal and run:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

4. Open the local URL displayed in your terminal.

## Project Limitations

- Churn probabilities are estimates, not certainties.
- Historical churn reasons describe customers who already left.
- Recorded reasons do not establish causation.
- Suggested retention actions require business validation.
- Campaign ROI results depend on user-provided assumptions.
- Synthetic demonstration results must not be presented as
  real customer findings.
- Model performance should be evaluated on held-out data.

## Future Improvements

- Explainable machine-learning predictions
- Improved customer-level reason reporting
- Model threshold tuning and calibration
- Retention campaign experiments
- Online dashboard deployment
- Monitoring for data drift

## Author

**Name:** Add your name here

**GitHub:** Add your profile URL here

## Disclaimer

Educational portfolio project. Validate dataset permissions,
model quality, privacy and business assumptions before real-world use.
