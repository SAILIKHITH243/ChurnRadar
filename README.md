# ChurnRadar
## Customer Churn Prediction and Retention Analytics Using Machine Learning

ChurnRadar is a Python-based application that analyses customer behaviour, estimates churn risk, explores recorded reasons why customers have already left, and supports customer-retention planning.

## Project Links

- **GitHub Repository:** [SAILIKHITH243/ChurnRadar](https://github.com/SAILIKHITH243/ChurnRadar)
- **Author Profile:** [Sai Likhith on GitHub](https://github.com/SAILIKHITH243)
- **Live Application:** Deployment link will be added after the public Streamlit app is successfully verified.

## Features

- Interactive customer analytics dashboard
- Machine-learning churn risk prediction
- Customer risk prioritisation
- Historical churn reason analysis
- Business insights from recorded churn reasons
- Model-based explanations of prediction factors
- Suggested customer-retention actions
- Retention campaign cost and ROI scenario simulator
- Model performance evaluation
- Downloadable CSV reports

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Streamlit
- Plotly
- Git and GitHub

## Dataset

This project uses a sample telecommunications customer churn dataset containing customer information and historical churn reasons.

Dataset source: [Telco Customer Churn on Hugging Face](https://huggingface.co/datasets/aai510-group1/telco-customer-churn)

The local preparation output reported 7,043 rows and 52 columns. This is sample data for educational analysis and does not represent live company customer records.

## Installation and Usage

1. Install Python.
2. Open the ChurnRadar folder in Visual Studio Code.
3. Open a PowerShell terminal.
4. Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Run the application:

```powershell
python -m streamlit run app.py
```

6. Open the local URL printed in the terminal.

## Business Applications

ChurnRadar can help teams explore historical churn patterns, identify customers for review, understand recorded reasons for customer departure, and compare potential retention campaign scenarios. Possible application areas include telecommunications, subscription services, FinTech, and marketing analytics.

## Limitations

- Churn probabilities are estimates, not certainties.
- Historical churn reasons describe customers who have already left.
- Model-based explanations do not prove the actual reason a customer may leave.
- Suggested retention actions require business validation.
- ROI simulator outputs depend on user-provided assumptions.
- Model performance must be evaluated on held-out data.
- Dataset redistribution terms should be checked before publishing raw data.

## Future Improvements

- Improved customer-level explanations
- Customer feedback and complaint analysis
- Model calibration and performance monitoring
- Retention campaign experiments
- Public application deployment and monitoring

## Author

**Sai Likhith**

GitHub Profile: [https://github.com/SAILIKHITH243](https://github.com/SAILIKHITH243)

## Disclaimer

This project was developed for educational and portfolio purposes. Predictions and recommendations should be validated before real-world use.
