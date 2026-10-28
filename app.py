from pathlib import Path
import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import plotly.io as pio

pio.templates["churnradar"] = go.layout.Template(
    layout=go.Layout(
        font=dict(family="Arial, sans-serif", color="#172554", size=13),
        title=dict(font=dict(color="#172554", size=18)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=["#0F766E", "#E8792E", "#14532D", "#4169A1", "#9F4A38"],
        xaxis=dict(
            color="#334155",
            title_font=dict(color="#172554"),
            tickfont=dict(color="#475569"),
            gridcolor="#E7E5E4",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1"
        ),
        yaxis=dict(
            color="#334155",
            title_font=dict(color="#172554"),
            tickfont=dict(color="#475569"),
            gridcolor="#E7E5E4",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1"
        ),
        legend=dict(font=dict(color="#172554")),
        margin=dict(t=65, b=55, l=55, r=25)
    )
)
px.defaults.template = "churnradar"


px.defaults.template = "plotly_white"

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report
)

st.set_page_config(
    page_title="ChurnRadar | Customer Intelligence",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY = "#142D4E"
TEAL = "#087F8C"
CORAL = "#E76F51"
AMBER = "#F4B942"
PALE = "#EAF2F8"
TEXT = "#243447"

st.markdown("""
<style>
.stApp,
[data-testid="stAppViewContainer"] {
    background: #FFFDF7 !important;
    color: #172554 !important;
}
[data-testid="stHeader"] {
    background: #FFFDF7 !important;
}
[data-testid="stSidebar"] {
    background: #172554 !important;
}
[data-testid="stSidebar"] * {
    color: #FFFFFF !important;
}
[data-testid="stMain"] h1,
[data-testid="stMain"] h2,
[data-testid="stMain"] h3,
[data-testid="stMain"] h4 {
    color: #172554 !important;
}
[data-testid="stMain"] p,
[data-testid="stMain"] label {
    color: #334155 !important;
}
[data-testid="stMetric"] {
    background: #FFFFFF !important;
    border: 1px solid #E7E5E4 !important;
    border-top: 4px solid #0F766E !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 4px 16px rgba(23,37,84,.06);
}
[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] p {
    color: #475569 !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] div,
[data-testid="stMetricValue"] span {
    color: #172554 !important;
}
[data-testid="stPlotlyChart"] {
    background: #FFFFFF !important;
    border: 1px solid #E7E5E4;
    border-radius: 14px;
    padding: 6px;
}
[data-testid="stFileUploader"] {
    background: #FFFFFF !important;
    border-radius: 12px;
    padding: 10px;
}
div.stButton > button {
    background: #0F766E !important;
    color: #FFFFFF !important;
    border: 0 !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
}
div.stButton > button:hover {
    background: #14532D !important;
    color: #FFFFFF !important;
}
.hero {
    background: linear-gradient(120deg, #172554 0%, #14532D 65%, #0F766E 100%) !important;
    padding: 30px;
    border-radius: 18px;
    color: #FFFFFF !important;
    margin-bottom: 22px;
}
.hero h1 {
    color: #FFFFFF !important;
    margin: 0;
    font-size: 34px;
}
.hero p {
    color: #F0FDF4 !important;
    margin: 8px 0 0 0;
}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------
# Demo data generator: clearly labelled synthetic data
# ------------------------------------------
@st.cache_data
def make_demo_data(n=2500):
    rng = np.random.default_rng(42)

    tenure = rng.integers(1, 73, n)
    monthly = np.round(rng.uniform(499, 2499, n), 2)
    usage = rng.poisson(18, n)
    satisfaction = rng.integers(1, 6, n)
    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        n, p=[0.55, 0.28, 0.17]
    )
    payment = rng.choice(
        ["Card", "Bank transfer", "Digital wallet"],
        n, p=[0.38, 0.32, 0.30]
    )

    logit = (
        1.0
        - 0.045 * tenure
        + 0.0018 * (monthly - 1500)
        - 0.075 * usage
        + 0.30 * (5 - satisfaction)
        + 1.0 * (contract == "Month-to-month")
        - 0.55 * (contract == "Two year")
    )
    probability = 1 / (1 + np.exp(-logit))
    churn = rng.binomial(1, probability)

    return pd.DataFrame({
        "CustomerID": [f"CR{i:05d}" for i in range(1, n + 1)],
        "TenureMonths": tenure,
        "MonthlyCharges": monthly,
        "MonthlyUsage": usage,
        "SatisfactionScore": satisfaction,
        "Contract": contract,
        "PaymentMethod": payment,
        "Churn": churn
    })


FEATURES = [
    "TenureMonths", "MonthlyCharges", "MonthlyUsage",
    "SatisfactionScore", "Contract", "PaymentMethod"
]
TARGET = "Churn"


@st.cache_resource
def train_models(df):
    numeric = [
        "TenureMonths", "MonthlyCharges",
        "MonthlyUsage", "SatisfactionScore"
    ]
    categorical = ["Contract", "PaymentMethod"]

    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=180, random_state=42,
            class_weight="balanced", min_samples_leaf=3
        )
    }

    results = {}
    for name, estimator in models.items():
        pipe = Pipeline([
            ("preprocess", preprocessor),
            ("model", estimator)
        ])
        pipe.fit(X_train, y_train)
        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]
        results[name] = {
            "pipeline": pipe,
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "auc": roc_auc_score(y_test, prob),
            "y_test": y_test,
            "pred": pred,
            "prob": prob
        }

    return results


# ------------------------------------------
# Data source
# ------------------------------------------
if "uploaded_df" not in st.session_state:
    st.session_state.uploaded_df = None

if "risk_df" not in st.session_state:
    st.session_state.risk_df = None

with st.sidebar:
    st.markdown("## 🎯 ChurnRadar")
    st.caption("Customer Intelligence Platform")
    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Customer Analytics",
            "Observed Churn Reasons",
            "Churn Prediction",
            "Risk Intelligence",
            "Churn Reasons & Explainable AI",
            "Campaign ROI Simulator",
            "Model Performance"
        ]
    )

    st.markdown("---")
    st.markdown("### Data source")
    upload = st.file_uploader(
        "Optional: upload a compatible CSV",
        type=["csv"]
    )
    st.caption(
        "CSV must contain the six model features. "
        "For analysis and training, include a binary Churn column."
    )

    if upload is not None:
        try:
            uploaded = pd.read_csv(upload)
            missing_features = [
                c for c in FEATURES if c not in uploaded.columns
            ]
            if missing_features:
                st.error("Missing columns: " + ", ".join(missing_features))
            else:
                st.session_state.uploaded_df = uploaded
                st.success("CSV loaded.")
        except Exception as exc:
            st.error(f"Could not read CSV: {exc}")

    if st.button("Use demo dataset", use_container_width=True):
        st.session_state.uploaded_df = None
        st.session_state.risk_df = None
        st.rerun()

if st.session_state.uploaded_df is not None:
    raw_df = st.session_state.uploaded_df.copy()
    data_source = "Uploaded CSV"
elif Path("data/churn_reasons_dataset.csv").exists():
    raw_df = pd.read_csv("data/churn_reasons_dataset.csv")
    data_source = "Extended telecom sample dataset with recorded reasons"
else:
    raw_df = make_demo_data()
    data_source = "Synthetic demonstration data"

# Map extended dataset columns to the model's expected feature names.
aliases = {
    "Customer ID": "CustomerID",
    "Tenure in Months": "TenureMonths",
    "Monthly Charge": "MonthlyCharges",
    "Avg Monthly GB Download": "MonthlyUsage",
    "Satisfaction Score": "SatisfactionScore",
    "Payment Method": "PaymentMethod"
}
for old_name, new_name in aliases.items():
    if old_name in raw_df.columns and new_name not in raw_df.columns:
        raw_df = raw_df.rename(columns={old_name: new_name})

# Standardise target values when supplied
if TARGET in raw_df.columns:
    if raw_df[TARGET].dtype == object:
        mapping = {
            "yes": 1, "no": 0, "true": 1, "false": 0,
            "1": 1, "0": 0, "churned": 1, "retained": 0
        }
        raw_df[TARGET] = (
            raw_df[TARGET].astype(str).str.strip().str.lower().map(mapping)
        )

    raw_df[TARGET] = pd.to_numeric(raw_df[TARGET], errors="coerce")

# Ensure features have expected data types
for c in ["TenureMonths", "MonthlyCharges", "MonthlyUsage", "SatisfactionScore"]:
    raw_df[c] = pd.to_numeric(raw_df[c], errors="coerce")

for c in ["Contract", "PaymentMethod"]:
    raw_df[c] = raw_df[c].astype("string").fillna("Unknown").astype(str)

# Training requires labelled records
labelled = raw_df.dropna(subset=[TARGET]).copy() if TARGET in raw_df.columns else pd.DataFrame()
if not labelled.empty:
    labelled = labelled[labelled[TARGET].isin([0, 1])].copy()
    labelled[TARGET] = labelled[TARGET].astype(int)

if not labelled.empty and labelled[TARGET].nunique() == 2:
    model_results = train_models(labelled)
    best_name = max(model_results, key=lambda k: model_results[k]["auc"])
    best_model = model_results[best_name]["pipeline"]
else:
    model_results = {}
    best_name = None
    best_model = None

# Clean display data
if "CustomerID" not in raw_df.columns:
    raw_df["CustomerID"] = [f"CR{i:05d}" for i in range(1, len(raw_df) + 1)]


def banner(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )



def show_chart(fig):
    # Force readable text and a proper light chart theme.
    fig.update_layout(
        template="plotly_white",
        font=dict(
            family="Arial, sans-serif",
            size=13,
            color="#172554"
        ),
        title=dict(font=dict(color="#172554", size=18)),
        paper_bgcolor="#FFFFFF",
        plot_bgcolor="#FFFFFF",
        colorway=[
            "#0F766E",
            "#E8792E",
            "#172554",
            "#16A34A",
            "#E76F51"
        ],
        xaxis=dict(
            tickfont=dict(color="#334155", size=12),
            title_font=dict(color="#172554", size=13),
            gridcolor="#E7E5E4",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1"
        ),
        yaxis=dict(
            tickfont=dict(color="#334155", size=12),
            title_font=dict(color="#172554", size=13),
            gridcolor="#E7E5E4",
            linecolor="#CBD5E1",
            zerolinecolor="#CBD5E1"
        ),
        legend=dict(font=dict(color="#172554"))
    )
    st.plotly_chart(fig, use_container_width=True)


def add_risk_scores(frame, model):
    result = frame.copy()
    result["ChurnProbability"] = model.predict_proba(result[FEATURES])[:, 1]
    result["RiskLevel"] = pd.cut(
        result["ChurnProbability"],
        bins=[-0.001, 0.30, 0.60, 1.001],
        labels=["Low", "Medium", "High"]
    ).astype(str)
    result["SuggestedAction"] = result["RiskLevel"].map({
        "Low": "Continue regular engagement",
        "Medium": "Review usage and offer helpful guidance",
        "High": "Prioritise a support review"
    })
    return result




def explain_customer_features(row, model, reference, features):
    """
    Show one-feature-at-a-time what-if sensitivity.
    These are model explanations, not confirmed real-world causes.
    """
    baseline = float(model.predict_proba(row[features])[0, 1])
    records = []

    for feature in features:
        altered = row[features].copy()
        idx = altered.index[0]
        altered.at[idx, feature] = reference[feature]

        changed_probability = float(
            model.predict_proba(altered)[0, 1]
        )

        impact = (baseline - changed_probability) * 100.0

        records.append({
            "Feature": feature,
            "Customer value": row.iloc[0][feature],
            "Typical value": reference[feature],
            "Current risk (%)": baseline * 100.0,
            "Risk if typical (%)": changed_probability * 100.0,
            "Impact (pp)": impact
        })

    result = pd.DataFrame(records)
    result["Direction"] = np.select(
        [
            result["Impact (pp)"] > 0.05,
            result["Impact (pp)"] < -0.05
        ],
        [
            "Raises model score",
            "Lowers model score"
        ],
        default="Little change"
    )

    return baseline, result.sort_values(
        "Impact (pp)", ascending=False
    )


def retention_action(feature, actual, typical):
    """Offer a review suggestion based on available data."""
    try:
        current = float(actual)
        reference_value = float(typical)
    except (ValueError, TypeError):
        current = None
        reference_value = None

    if feature == "SatisfactionScore" and current is not None:
        if current < reference_value:
            return "Review customer feedback and investigate dissatisfaction."
    if feature == "MonthlyUsage" and current is not None:
        if current < reference_value:
            return "Review engagement and check whether the service meets the customer's needs."
    if feature == "TenureMonths" and current is not None:
        if current < reference_value:
            return "Review onboarding and the early customer experience."
    if feature == "MonthlyCharges" and current is not None:
        if current > reference_value:
            return "Review plan suitability, pricing clarity and perceived value."
    if feature == "Contract":
        return "Review contract flexibility and explain available plan options."
    if feature == "PaymentMethod":
        return "Check for payment friction and whether suitable payment options are available."
    return "Review the customer's available history before deciding on an action."

# ------------------------------------------
# Executive Overview
# ------------------------------------------
if page == "Executive Overview":
    banner(
        "Customer Intelligence Overview",
        "Understand customer behaviour, churn patterns and estimated risk."
    )

    c1, c2, c3, c4 = st.columns(4)
    total = len(raw_df)
    observed = labelled[TARGET].mean() * 100 if not labelled.empty else None
    risk_data = add_risk_scores(raw_df, best_model) if best_model else None

    c1.metric("Total customers", f"{total:,}")
    c2.metric(
        "Observed churn",
        f"{observed:.1f}%" if observed is not None else "Unavailable"
    )
    c3.metric(
        "High-risk estimates",
        f"{(risk_data['RiskLevel'] == 'High').sum():,}"
        if risk_data is not None else "Train a model"
    )
    c4.metric("Selected model", best_name or "Unavailable")

    left, right = st.columns(2)
    with left:
        st.subheader("Customer contract mix")
        fig = px.histogram(
            raw_df, x="Contract", color="Contract",
            title="Customers by contract type",
            color_discrete_sequence=[TEAL, NAVY, AMBER]
        )
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)

    with right:
        st.subheader("Customer tenure")
        fig = px.histogram(
            raw_df, x="TenureMonths", nbins=24,
            title="Distribution of tenure",
            color_discrete_sequence=[TEAL]
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)

    st.subheader("Churn by contract")
    if not labelled.empty:
        churn_by_contract = (
            labelled.groupby("Contract", observed=False)[TARGET]
            .mean().mul(100).reset_index(name="ChurnRate")
        )
        fig = px.bar(
            churn_by_contract, x="Contract", y="ChurnRate",
            text_auto=".1f", color="ChurnRate",
            color_continuous_scale=["#DDF4EF", TEAL, CORAL],
            labels={"ChurnRate": "Observed churn (%)"}
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)
    else:
        st.info("Upload labelled data with a Churn column to analyse observed churn.")

    st.caption(
        f"Data source: {data_source}. "
        "If using demo data, records are synthetic and are not real customers."
    )


# ------------------------------------------
# Customer Analytics
# ------------------------------------------
elif page == "Customer Analytics":
    banner(
        "Customer Analytics",
        "Explore customer characteristics and compare available segments."
    )

    contract_options = sorted(raw_df["Contract"].dropna().unique().tolist())
    chosen_contracts = st.multiselect(
        "Filter by contract", contract_options, default=contract_options
    )
    view = raw_df[raw_df["Contract"].isin(chosen_contracts)].copy()

    a, b, c = st.columns(3)
    a.metric("Filtered customers", f"{len(view):,}")
    b.metric("Average tenure", f"{view['TenureMonths'].mean():.1f} months")
    c.metric("Average monthly charge", f"₹{view['MonthlyCharges'].mean():,.2f}")

    left, right = st.columns(2)
    with left:
        fig = px.box(
            view, x="Contract", y="MonthlyCharges", color="Contract",
            title="Monthly charges by contract (₹)",
            color_discrete_sequence=[TEAL, NAVY, AMBER]
        )
        fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)

    with right:
        fig = px.scatter(
            view, x="TenureMonths", y="MonthlyUsage",
            color="Contract", size="MonthlyCharges",
            hover_data=["CustomerID"],
            title="Tenure vs monthly usage"
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)

    if not labelled.empty:
        st.subheader("Observed churn by customer segment")
        segment = (
            labelled.groupby("Contract", observed=False)[TARGET]
            .agg(["mean", "count"]).reset_index()
        )
        segment["ChurnRatePct"] = segment["mean"] * 100
        segment = segment.rename(columns={"count": "Customers"})
        st.dataframe(
            segment[["Contract", "Customers", "ChurnRatePct"]],
            use_container_width=True, hide_index=True
        )

    st.download_button(
        "Download filtered customer data",
        data=view.to_csv(index=False).encode("utf-8"),
        file_name="churnradar_customer_analysis.csv",
        mime="text/csv"
    )


# ------------------------------------------
# Churn Prediction
# ------------------------------------------
elif page == "Churn Prediction":
    banner(
        "Churn Prediction",
        "Score customers using a model trained on labelled historical records."
    )

    if best_model is None:
        st.warning(
            "A prediction model requires a CSV with all six feature columns "
            "and a valid binary Churn column containing both classes."
        )
    else:
        st.success(f"Ready to predict. Selected model: {best_name}")
        st.write("You can score the currently loaded customer records or upload a compatible CSV in the sidebar.")

        prediction_input = raw_df.copy()
        if st.button("Predict churn risk for loaded customers", type="primary"):
            st.session_state.risk_df = add_risk_scores(prediction_input, best_model)

        if st.session_state.risk_df is not None:
            scored = st.session_state.risk_df
            st.dataframe(
                scored[[
                    "CustomerID", "TenureMonths", "MonthlyCharges",
                    "Contract", "ChurnProbability", "RiskLevel"
                ]].sort_values("ChurnProbability", ascending=False),
                use_container_width=True, hide_index=True
            )
            st.download_button(
                "Download predictions",
                data=scored.to_csv(index=False).encode("utf-8"),
                file_name="churnradar_predictions.csv",
                mime="text/csv"
            )


# ------------------------------------------
# Risk Intelligence
# ------------------------------------------
elif page == "Risk Intelligence":
    banner(
        "Customer Risk Intelligence",
        "Prioritise customers for review using estimated churn probability."
    )

    if best_model is None:
        st.warning("Load labelled data with both churn outcomes to train the model.")
    else:
        scored = add_risk_scores(raw_df, best_model)
        high = scored[scored["RiskLevel"] == "High"]
        medium = scored[scored["RiskLevel"] == "Medium"]
        low = scored[scored["RiskLevel"] == "Low"]

        a, b, c = st.columns(3)
        a.metric("High risk", f"{len(high):,}")
        b.metric("Medium risk", f"{len(medium):,}")
        c.metric("Low risk", f"{len(low):,}")

        fig = px.histogram(
            scored, x="ChurnProbability", nbins=25,
            title="Distribution of predicted churn probabilities",
            color_discrete_sequence=[TEAL]
        )
        fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
        show_chart(fig)

        risk_filter = st.multiselect(
            "Risk groups", ["High", "Medium", "Low"],
            default=["High", "Medium", "Low"]
        )
        filtered = scored[scored["RiskLevel"].isin(risk_filter)].copy()
        filtered = filtered.sort_values("ChurnProbability", ascending=False)

        st.dataframe(
            filtered[[
                "CustomerID", "TenureMonths", "MonthlyUsage",
                "SatisfactionScore", "ChurnProbability",
                "RiskLevel", "SuggestedAction"
            ]],
            use_container_width=True, hide_index=True
        )
        st.download_button(
            "Export prioritised customer list",
            data=filtered.to_csv(index=False).encode("utf-8"),
            file_name="churnradar_risk_priorities.csv",
            mime="text/csv"
        )
        st.caption(
            "Risk bands are illustrative configurable thresholds. "
            "Suggested actions are prompts for human review, not guaranteed interventions."
        )


# ------------------------------------------
# Campaign ROI Simulator
# ------------------------------------------
elif page == "Campaign ROI Simulator":
    banner(
        "Retention Campaign Simulator",
        "Explore hypothetical campaign costs and break-even assumptions."
    )

    st.info(
        "This is a scenario calculator, not a prediction that a campaign will "
        "actually prevent churn. Enter your own assumptions."
    )

    left, right = st.columns(2)
    with left:
        contacted = st.number_input(
            "Customers to contact", min_value=0, max_value=1000000,
            value=500, step=50
        )
        contact_cost = st.number_input(
            "Contact cost per customer (₹)", min_value=0.0,
            value=10.0, step=5.0
        )
        incentive_cost = st.number_input(
            "Incentive cost per contacted customer (₹)",
            min_value=0.0, value=40.0, step=10.0
        )
    with right:
        success_pct = st.slider(
            "Assumed successful retention rate (%)",
            min_value=0, max_value=100, value=10
        )
        contribution = st.number_input(
            "Estimated contribution per retained customer (₹)",
            min_value=0.0, value=500.0, step=50.0
        )

    campaign_cost = contacted * (contact_cost + incentive_cost)
    retained = contacted * success_pct / 100
    contribution_value = retained * contribution
    net_benefit = contribution_value - campaign_cost

    a, b, c = st.columns(3)
    a.metric("Campaign cost", f"₹{campaign_cost:,.2f}")
    b.metric("Estimated retained customers", f"{retained:,.1f}")
    c.metric("Illustrative net benefit", f"₹{net_benefit:,.2f}")

    if contacted > 0 and contribution > 0:
        break_even = (campaign_cost / (contacted * contribution)) * 100
        st.metric("Break-even retention rate", f"{break_even:.1f}%")
    else:
        st.metric("Break-even retention rate", "Not available")

    sim = pd.DataFrame({
        "Metric": ["Campaign cost", "Estimated contribution", "Illustrative net benefit"],
        "Amount (₹)": [campaign_cost, contribution_value, net_benefit]
    })
    fig = px.bar(
        sim, x="Metric", y="Amount (₹)",
        color="Metric",
        color_discrete_sequence=[NAVY, TEAL, CORAL],
        title="Campaign scenario summary"
    )
    fig.update_layout(showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
    show_chart(fig)

    st.caption(
        "Illustrative estimates only. Contribution is not the same as revenue. "
        "Actual campaign impact should be measured with a controlled experiment."
    )


# ------------------------------------------
# Model Performance
# ------------------------------------------

elif page == "Observed Churn Reasons":
    banner(
        "Observed Churn Reasons",
        "Explore reasons recorded for customers who have already churned."
    )

    if "Churn Reason" not in raw_df.columns:
        st.warning(
            "This dataset does not contain recorded churn reasons. "
            "Load the extended telecom dataset to use this page."
        )
    else:
        observed = raw_df[
            pd.to_numeric(raw_df[TARGET], errors="coerce") == 1
        ].copy()

        observed["Churn Reason"] = (
            observed["Churn Reason"].fillna("").astype(str).str.strip()
        )
        observed = observed[
            observed["Churn Reason"].ne("")
            & observed["Churn Reason"].str.lower().ne("nan")
        ].copy()

        if observed.empty:
            st.info("No recorded churn reasons were found in this dataset.")
        else:
            if "Churn Category" not in observed.columns:
                observed["Churn Category"] = "Uncategorised"

            observed["Churn Category"] = (
                observed["Churn Category"].fillna("Uncategorised").astype(str)
            )

            def action_for_reason(category, reason):
                text = (str(category) + " " + str(reason)).lower()

                if any(word in text for word in ["competitor", "better offer", "better device"]):
                    return "Review competitive offers and whether the current service meets customer needs."
                if any(word in text for word in ["price", "cost", "expensive"]):
                    return "Review pricing clarity, plan suitability and available lower-cost options."
                if any(word in text for word in ["support", "attitude", "representative", "expertise"]):
                    return "Review customer support records, response time and unresolved complaints."
                if any(word in text for word in ["network", "reliability", "service", "outage"]):
                    return "Investigate service reliability and check for repeated technical incidents."
                if any(word in text for word in ["dissatisfaction", "product", "quality"]):
                    return "Review customer feedback and investigate possible product or service problems."
                if any(word in text for word in ["relocat", "moved"]):
                    return "Review service availability and explain any suitable alternatives."
                return "Review the recorded feedback and customer history before deciding on an action."

            a, b, c = st.columns(3)
            a.metric("Customers with recorded reasons", f"{len(observed):,}")
            b.metric("Distinct reasons", f"{observed['Churn Reason'].nunique():,}")
            c.metric("Reason categories", f"{observed['Churn Category'].nunique():,}")

            st.caption(
                "These are historical reasons recorded in a sample telecom dataset. "
                "They are not confirmed reasons for customers currently at risk."
            )

            left, right = st.columns(2)

            with left:
                category_counts = (
                    observed["Churn Category"]
                    .value_counts()
                    .rename_axis("Category")
                    .reset_index(name="Customers")
                )
                fig = px.bar(
                    category_counts,
                    x="Category",
                    y="Customers",
                    color="Category",
                    title="Recorded churn reasons by category",
                    color_discrete_sequence=[
                        "#0F766E", "#E8792E", "#14532D",
                        "#4169A1", "#9F4A38"
                    ]
                )
                if "show_chart" in globals():
                    show_chart(fig)
                else:
                    st.plotly_chart(fig, use_container_width=True)

            with right:
                reason_counts = (
                    observed["Churn Reason"]
                    .value_counts()
                    .head(10)
                    .rename_axis("Reason")
                    .reset_index(name="Customers")
                )
                fig = px.bar(
                    reason_counts.sort_values("Customers"),
                    x="Customers",
                    y="Reason",
                    orientation="h",
                    title="Top 10 recorded reasons",
                    color_discrete_sequence=["#0F766E"]
                )
                if "show_chart" in globals():
                    show_chart(fig)
                else:
                    st.plotly_chart(fig, use_container_width=True)

            st.subheader("Business insights from recorded reasons")

            reason_text = (
                observed["Churn Reason"]
                .fillna("")
                .astype(str)
                .str.lower()
            )
            total_reasons = len(observed)

            competitor_count = int(
                reason_text.str.contains(
                    r"competitor|better offer|better devices",
                    regex=True, na=False
                ).sum()
            )
            support_count = int(
                reason_text.str.contains(
                    r"support|service provider|attitude",
                    regex=True, na=False
                ).sum()
            )
            price_count = int(
                reason_text.str.contains(
                    r"price|cost|expensive|billing",
                    regex=True, na=False
                ).sum()
            )
            network_count = int(
                reason_text.str.contains(
                    r"network|reliability|outage|download speed",
                    regex=True, na=False
                ).sum()
            )

            col1, col2, col3, col4 = st.columns(4)

            insight_items = [
                (col1, "Competitor-related", competitor_count),
                (col2, "Support / service", support_count),
                (col3, "Pricing-related", price_count),
                (col4, "Network / reliability", network_count)
            ]

            for column, label, count in insight_items:
                with column:
                    st.metric(label, f"{count:,}")
                    pct = count / total_reasons * 100 if total_reasons else 0
                    st.caption(f"{pct:.1f}% of recorded churn reasons")

            top_reasons = observed["Churn Reason"].value_counts()

            if not top_reasons.empty:
                top_reason = str(top_reasons.index[0])
                top_count = int(top_reasons.iloc[0])
                top_pct = (
                    top_count / total_reasons * 100
                    if total_reasons else 0
                )
                st.info(
                    f"Most frequent recorded reason: {top_reason} "
                    f"({top_count:,} records, {top_pct:.1f}% of records "
                    "with a recorded reason)."
                )

            st.subheader("Possible business actions")

            if competitor_count > 0:
                st.write(
                    "- **Competition:** Compare the company's offers, "
                    "features and service options with alternatives."
                )
            if support_count > 0:
                st.write(
                    "- **Customer support:** Review complaint handling, "
                    "staff interactions and unresolved issues."
                )
            if price_count > 0:
                st.write(
                    "- **Pricing:** Review plan suitability, price clarity "
                    "and whether customers understand available options."
                )
            if network_count > 0:
                st.write(
                    "- **Service reliability:** Investigate relevant "
                    "network, speed or reliability complaints."
                )

            st.caption(
                "Groups are based on keywords in recorded reasons and may "
                "overlap. They are exploratory summaries, not proof of cause. "
                "Percentages use records with a recorded churn reason."
            )

            st.subheader("Explore a recorded customer reason")
            reason_filter = st.multiselect(
                "Filter by reason category",
                sorted(observed["Churn Category"].unique()),
                default=sorted(observed["Churn Category"].unique())
            )
            filtered = observed[
                observed["Churn Category"].isin(reason_filter)
            ].copy()

            id_col = "CustomerID" if "CustomerID" in filtered.columns else (
                "Customer ID" if "Customer ID" in filtered.columns else None
            )

            display_cols = [
                c for c in [
                    id_col, "Churn Category", "Churn Reason",
                    "Contract", "Satisfaction Score", "Monthly Charge"
                ]
                if c and c in filtered.columns
            ]

            st.dataframe(
                filtered[display_cols],
                use_container_width=True,
                hide_index=True
            )

            if not filtered.empty:
                if id_col:
                    selected_id = st.selectbox(
                        "Choose a historical churned customer",
                        filtered[id_col].astype(str).unique().tolist()
                    )
                    selected = filtered[
                        filtered[id_col].astype(str) == selected_id
                    ].iloc[0]
                    st.markdown("### Recorded reason")
                    st.write(f"**Customer:** {selected_id}")
                    st.write(f"**Category:** {selected['Churn Category']}")
                    st.write(f"**Recorded reason:** {selected['Churn Reason']}")
                    st.write(
                        "**Possible business action:** "
                        + action_for_reason(
                            selected["Churn Category"],
                            selected["Churn Reason"]
                        )
                    )

                st.download_button(
                    "Download recorded churn reasons",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="churnradar_recorded_churn_reasons.csv",
                    mime="text/csv"
                )

elif page == "Churn Reasons & Explainable AI":
    banner(
        "Customer Churn Reasons & Explainable AI",
        "Explore the factors that influence each customer's estimated churn risk."
    )

    if best_model is None or labelled.empty:
        st.warning(
            "A trained model is required. Load labelled data containing "
            "both churn outcomes and the required model features."
        )
    else:
        st.info(
            "These are model-based what-if explanations, not confirmed reasons why "
            "a customer will leave. Each feature is changed individually "
            "to a typical reference value to estimate its local influence."
        )

        scored = add_risk_scores(raw_df, best_model)
        customer_ids = (
            scored["CustomerID"].astype(str).drop_duplicates().tolist()
        )

        selected_id = st.selectbox(
            "Select a customer", customer_ids
        )

        customer_row = scored.loc[
            scored["CustomerID"].astype(str) == selected_id,
            FEATURES
        ].head(1).copy()

        numeric_features = [
            "TenureMonths", "MonthlyCharges",
            "MonthlyUsage", "SatisfactionScore"
        ]

        reference = {}
        for feature in FEATURES:
            if feature in numeric_features:
                reference[feature] = pd.to_numeric(
                    labelled[feature], errors="coerce"
                ).median()
            else:
                values = labelled[feature].dropna().astype(str)
                reference[feature] = (
                    values.mode().iloc[0] if not values.empty else "Unknown"
                )

        risk, explanations = explain_customer_features(
            customer_row, best_model, reference, FEATURES
        )

        risk_col, level_col, model_col = st.columns(3)
        risk_col.metric("Estimated churn risk", f"{risk * 100:.1f}%")
        risk_level = (
            "High" if risk >= 0.60
            else "Medium" if risk >= 0.30
            else "Low"
        )
        level_col.metric("Risk category", risk_level)
        model_col.metric("Model used", best_name or "Unavailable")

        feature_labels = {
            "TenureMonths": "Customer tenure",
            "MonthlyCharges": "Monthly charges",
            "MonthlyUsage": "Monthly usage",
            "SatisfactionScore": "Customer satisfaction score",
            "Contract": "Contract type",
            "PaymentMethod": "Payment method"
        }

        risk_raisers = explanations[
            explanations["Impact (pp)"] > 0.05
        ].head(3)

        st.subheader("Factors influencing the model score")

        if risk_raisers.empty:
            st.info(
                "No individual feature increased risk relative to its "
                "typical reference value. The strongest local influences "
                "are shown below, including factors that lower risk."
            )
            top_factors = (
                explanations.assign(
                    _magnitude=explanations["Impact (pp)"].abs()
                )
                .sort_values("_magnitude", ascending=False)
                .head(3)
            )
        else:
            top_factors = risk_raisers

        for _, factor in top_factors.iterrows():
            feature = str(factor["Feature"])
            actual = factor["Customer value"]
            typical = factor["Typical value"]
            impact = float(factor["Impact (pp)"])
            name = feature_labels.get(feature, feature)

            counterfactual_risk = float(factor["Risk if typical (%)"])

            if impact > 0.05:
                explanation_text = (
                    f"The current model score is {risk * 100:.1f}%. "
                    f"If only this feature changed to its typical reference "
                    f"({typical}), the model score would be "
                    f"{counterfactual_risk:.1f}%. That is a change of "
                    f"{impact:.1f} percentage points. This is a model "
                    "what-if comparison, not proof of the customer's actual reason."
                )
                action_text = retention_action(feature, actual, typical)
            elif impact < -0.05:
                explanation_text = (
                    f"The current model score is {risk * 100:.1f}%. "
                    f"If only this feature changed to its typical reference "
                    f"({typical}), the model score would be "
                    f"{counterfactual_risk:.1f}%. That is a change of "
                    f"{abs(impact):.1f} percentage points in the opposite "
                    "direction. This does not establish a real-world cause."
                )
                action_text = (
                    "No corrective action is inferred from this factor alone."
                )
            else:
                explanation_text = (
                    "Replacing this feature with its typical reference "
                    "makes very little difference to this prediction."
                )
                action_text = (
                    "Review the full customer context before deciding on an action."
                )

            with st.expander(name, expanded=True):
                st.write(f"**Customer value:** {actual}")
                st.write(f"**Typical reference value:** {typical}")
                st.write(explanation_text)
                st.caption(f"Possible follow-up: {action_text}")

        chart_data = explanations.copy()
        chart_data["Feature label"] = chart_data["Feature"].map(
            feature_labels
        )
        fig = px.bar(
            chart_data.sort_values("Impact (pp)"),
            x="Impact (pp)",
            y="Feature label",
            color="Direction",
            orientation="h",
            color_discrete_map={
                "Raises estimated risk": "#E76F51",
                "Lowers estimated risk": "#0F766E",
                "Little change": "#94A3B8"
            },
            title="How individual feature replacements influence the score",
            labels={
                "Impact (pp)": "Risk change (percentage points)",
                "Feature label": "Customer feature"
            }
        )
        fig.update_layout(
            paper_bgcolor="#FFFFFF",
            plot_bgcolor="#FFFFFF",
            font=dict(color="#172554"),
            title_font=dict(color="#172554")
        )

        if "show_chart" in globals():
            show_chart(fig)
        else:
            st.plotly_chart(fig, use_container_width=True)

        report = explanations.copy()
        report.insert(0, "CustomerID", selected_id)
        report.insert(1, "EstimatedChurnRiskPct", round(risk * 100, 2))
        report["Feature"] = report["Feature"].map(
            lambda value: feature_labels.get(value, value)
        )

        st.download_button(
            "Download customer explanation CSV",
            data=report.to_csv(index=False).encode("utf-8"),
            file_name=f"churnradar_explanation_{selected_id}.csv",
            mime="text/csv"
        )

elif page == "Model Performance":
    banner(
        "Model Performance",
        "Evaluate how well the models distinguish churned from retained customers."
    )

    if not model_results:
        st.warning("Upload labelled data with both churn outcomes to evaluate models.")
    else:
        comparison = pd.DataFrame([
            {
                "Model": name,
                "Accuracy": result["accuracy"],
                "Precision": result["precision"],
                "Recall": result["recall"],
                "F1-score": result["f1"],
                "ROC-AUC": result["auc"]
            }
            for name, result in model_results.items()
        ])
        st.dataframe(
            comparison.style.format({
                "Accuracy": "{:.3f}", "Precision": "{:.3f}",
                "Recall": "{:.3f}", "F1-score": "{:.3f}",
                "ROC-AUC": "{:.3f}"
            }),
            use_container_width=True, hide_index=True
        )

        selected = st.selectbox("Inspect model", list(model_results.keys()))
        result = model_results[selected]

        a, b, c, d = st.columns(4)
        a.metric("Precision", f"{result['precision']:.3f}")
        b.metric("Recall", f"{result['recall']:.3f}")
        c.metric("F1-score", f"{result['f1']:.3f}")
        d.metric("ROC-AUC", f"{result['auc']:.3f}")

        cm = confusion_matrix(result["y_test"], result["pred"], labels=[0, 1])
        fig = px.imshow(
            cm, text_auto=True, x=["Predicted retained", "Predicted churn"],
            y=["Actual retained", "Actual churn"],
            color_continuous_scale=["white", TEAL],
            title=f"Confusion matrix — {selected}"
        )
        fig.update_layout(paper_bgcolor="white")
        show_chart(fig)

        st.markdown("**How to interpret these metrics**")
        st.write("- **Precision:** Of customers flagged as churners, how many were churners in the test set?")
        st.write("- **Recall:** Of actual churners in the test set, how many did the model identify?")
        st.write("- **F1-score:** A balance between precision and recall.")
        st.write("- **ROC-AUC:** How well the model ranks churners above retained customers across thresholds.")
        st.caption(
            "These are held-out test results for this dataset and pipeline. "
            "They do not guarantee future business performance. Demo data is synthetic."
        )

st.markdown("---")
st.caption(
    "ChurnRadar | Customer Churn Prediction & Retention Analytics | "
    "Educational portfolio project. Validate data, fairness, privacy and "
    "retention outcomes before real-world use."
)
