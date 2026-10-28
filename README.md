# ChurnRadar
## Customer Churn Prediction and Retention Analytics Using Machine Learning

ChurnRadar is a machine learning application designed to analyse customer behaviour, predict churn risk, explore recorded churn reasons, and support customer retention decisions.

## Project Links

- GitHub Repository: https://github.com/SAILIKHITH243/ChurnRadar
- GitHub Profile: https://github.com/SAILIKHITH243
- Live Application: https://churnradar-rwvogsqrmveycb37t7dp3i.streamlit.app/

Note: The application URL must be verified after successful deployment.

## Project Features

- Interactive customer analytics dashboard
- Customer churn prediction using machine learning
- Customer risk prioritisation
- Historical churn reason analysis
- Churn category and business insights
- Model-based customer risk explanations
- Recommended customer retention actions
- Retention campaign cost and ROI simulator
- Model performance evaluation
- Downloadable CSV reports

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit
- Plotly
- Git
- GitHub

## Dataset

The project uses a sample telecommunications customer churn dataset containing customer information and historical churn reasons.

Dataset source:
https://huggingface.co/datasets/aai510-group1/telco-customer-churn

The prepared local dataset was reported to contain 7,043 rows and 52 columns. It is sample data for educational analysis and does not represent live company customer records.

## Installation and Usage

1. Install Python.
2. Clone or download the repository.
3. Open the project folder in Visual Studio Code.
4. Open a terminal and install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

5. Start the application:

```powershell
python -m streamlit run app.py
```

6. Open the local URL displayed in the terminal.

## Business Applications

ChurnRadar can help businesses understand historical customer churn, identify patterns in customer behaviour, prioritise customers for review, and evaluate possible retention campaign scenarios.

Potential applications include telecommunications, subscription services, FinTech, and customer marketing analytics.

## Limitations

- Churn probabilities are estimates, not certainties.
- Historical churn reasons describe customers who have already left.
- Model-based explanations do not establish the actual reason a customer may leave.
- Suggested retention actions require business validation.
- ROI simulator results depend on user-provided assumptions.
- Model performance must be evaluated using held-out test data.
- Dataset permissions should be checked before redistributing source data.

## Future Improvements

- Enhanced customer-level explanations
- Customer feedback and complaint analysis
- Improved model calibration and evaluation
- Retention campaign experiments
- Online application monitoring
- Model drift detection

## Author

Sai Likhith

GitHub Profile:
https://github.com/SAILIKHITH243

## Disclaimer

This project was developed for educational and portfolio purposes. Model predictions and business recommendations should be validated before real-world use.
