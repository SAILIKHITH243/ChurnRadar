# ChurnRadar
## Customer Churn Prediction and Retention Analytics Using Machine Learning

ChurnRadar is a Python-based application that analyses customer behaviour, predicts churn risk, and explores recorded reasons why customers have left a service.

## Project Links

- GitHub Repository: https://github.com/SAILIKHITH243/ChurnRadar
- Author Profile: https://github.com/SAILIKHITH243
- Live Application: Deployment pending verification

## Features

- Customer analytics dashboard
- Machine learning churn prediction
- Customer risk prioritisation
- Historical churn reason analysis
- Business insights from recorded churn reasons
- Model-based customer risk explanations
- Suggested retention actions
- Retention campaign cost and ROI simulator
- Model performance evaluation
- CSV report downloads

## Technologies

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Git and GitHub

## Dataset

The project uses a sample telecommunications customer churn dataset containing customer information and historical churn reasons.

Dataset source:
https://huggingface.co/datasets/aai510-group1/telco-customer-churn

The locally prepared dataset was reported to contain 7,043 rows and 52 columns. The data is for educational analysis and does not represent live company customer records.

## Installation and Usage

1. Install Python.
2. Open the ChurnRadar folder in Visual Studio Code.
3. Open a PowerShell terminal.
4. Install the required libraries:

```powershell
python -m pip install -r requirements.txt
```

5. Start the application:

```powershell
python -m streamlit run app.py
```

6. Open the local URL displayed in the terminal.

## Business Applications

ChurnRadar helps businesses explore historical churn patterns, identify customers for review, understand recorded reasons for customer departure, and evaluate possible retention campaign scenarios.

Potential application areas include telecommunications, subscription services, FinTech, and marketing analytics.

## Limitations

- Churn probabilities are estimates, not certainties.
- Historical reasons describe customers who have already left.
- Model-based explanations do not establish actual customer reasons.
- Suggested retention actions require business validation.
- ROI calculations depend on user-provided assumptions.
- Model performance must be evaluated on held-out data.
- Dataset redistribution permissions should be checked before publishing raw data.

## Future Improvements

- Improved customer-level explanations
- Customer feedback and complaint analysis
- Model calibration and performance monitoring
- Retention campaign experiments
- Online application deployment

## Author

**Sai Likhith**

GitHub Profile:
https://github.com/SAILIKHITH243

## Disclaimer

This project was developed for educational and portfolio purposes. Predictions and recommendations should be validated before real-world use.
