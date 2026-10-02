"""
Online Payment Fraud Detection — Streamlit App
---------------------------------------------------
Interactive web app that loads the trained model and lets a user enter
transaction details to get a live fraud-risk prediction, alongside
dataset EDA and model performance metrics.

Run with:
    streamlit run app.py
"""

import json

import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Online Payment Fraud Detector",
    page_icon="💳",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    model = joblib.load("model/fraud_model.pkl")
    scaler = joblib.load("model/scaler.pkl")
    with open("model/metadata.json") as f:
        metadata = json.load(f)
    return model, scaler, metadata


@st.cache_data
def load_dataset():
    return pd.read_csv("data/fraud.csv")


model, scaler, metadata = load_artifacts()
df = load_dataset()

encoders = metadata["encoders"]
feature_cols = metadata["feature_cols"]
uses_scaled_input = metadata["uses_scaled_input"]
amount_min, amount_max = metadata["amount_range"]

st.sidebar.title("💳 Fraud Detection")
page = st.sidebar.radio("Navigate", ["Predict", "Dataset Overview", "Model Performance"])
st.sidebar.markdown("---")
st.sidebar.caption(
    f"Deployed model: **{metadata['best_model']}**  \n"
    f"Fraud rate in data: **{(df['Fraud'] == 'Yes').mean():.2%}**"
)

# --------------------------------------------------------------------------
# PAGE 1 — Prediction
# --------------------------------------------------------------------------
if page == "Predict":
    st.title("Online Payment Fraud Risk Predictor")
    st.write(
        f"Enter transaction details to estimate fraud risk using a "
        f"**{metadata['best_model']}** model trained on {len(df):,} transactions."
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.slider("Customer Age", 18, 70, 35)
        amount = st.number_input(
            "Transaction Amount (₹)", min_value=float(amount_min), max_value=float(amount_max),
            value=float(round((amount_min + amount_max) / 4, 2)),
        )
        hour = st.slider("Transaction Hour (0–23)", 0, 23, 14)

    with col2:
        prev_txns = st.slider("Previous Transactions", 1, 50, 25)
        gender = st.selectbox("Gender", encoders["Gender"])
        payment_type = st.selectbox("Payment Type", encoders["Payment_Type"])

    with col3:
        location = st.selectbox("Location", encoders["Location"])
        device = st.selectbox("Device Type", encoders["Device_Type"])
        network = st.selectbox("Network Type", encoders["Network_Type"])

    threshold = st.slider(
        "Decision threshold (probability above which a transaction is flagged)",
        0.05, 0.95, 0.5, 0.05,
        help="Lower the threshold to catch more fraud at the cost of more false alarms.",
    )

    if st.button("Check Transaction", type="primary", use_container_width=True):
        input_dict = {
            "Age": age,
            "Transaction_Amount": amount,
            "Transaction_Hour": hour,
            "Previous_Transactions": prev_txns,
            "Gender": encoders["Gender"].index(gender),
            "Payment_Type": encoders["Payment_Type"].index(payment_type),
            "Location": encoders["Location"].index(location),
            "Device_Type": encoders["Device_Type"].index(device),
            "Network_Type": encoders["Network_Type"].index(network),
        }
        input_df = pd.DataFrame([input_dict])[feature_cols]
        input_array = scaler.transform(input_df) if uses_scaled_input else input_df

        probability = model.predict_proba(input_array)[0][1]
        flagged = probability >= threshold

        st.markdown("---")
        result_col, gauge_col = st.columns([1, 1])
        with result_col:
            if flagged:
                st.error(f"### 🚩 Flagged as Potential Fraud\nEstimated probability: **{probability:.1%}**")
            else:
                st.success(f"### ✅ Looks Legitimate\nEstimated probability: **{probability:.1%}**")
            st.caption("Demo model for portfolio purposes — not a production fraud system.")
        with gauge_col:
            fig = px.pie(
                values=[probability, 1 - probability],
                names=["Fraud Risk", "Legitimate"],
                hole=0.6,
                color_discrete_sequence=["#EF553B", "#636EFA"],
            )
            fig.update_layout(showlegend=False, margin=dict(t=0, b=0, l=0, r=0), height=250)
            st.plotly_chart(fig, use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 2 — Dataset Overview
# --------------------------------------------------------------------------
elif page == "Dataset Overview":
    st.title("Dataset Overview")
    st.write(f"**{len(df):,} transactions** · **{df.shape[1]} columns**")

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Transactions", f"{len(df):,}")
    m2.metric("Flagged Fraud", f"{(df['Fraud'] == 'Yes').sum():,}")
    m3.metric("Fraud Rate", f"{(df['Fraud'] == 'Yes').mean():.2%}")

    st.subheader("Class Balance (highly imbalanced — typical for real fraud data)")
    class_counts = df["Fraud"].value_counts().reset_index()
    class_counts.columns = ["Fraud", "Count"]
    st.plotly_chart(
        px.bar(class_counts, x="Fraud", y="Count", color="Fraud", text="Count", log_y=True),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Transaction Amount Distribution")
        st.plotly_chart(
            px.histogram(df, x="Transaction_Amount", color="Fraud", barmode="overlay", nbins=40),
            use_container_width=True,
        )
    with c2:
        st.subheader("Fraud Rate by Payment Type")
        rate_by_type = (
            df.assign(is_fraud=(df["Fraud"] == "Yes").astype(int))
            .groupby("Payment_Type")["is_fraud"].mean().reset_index()
        )
        st.plotly_chart(
            px.bar(rate_by_type, x="Payment_Type", y="is_fraud", labels={"is_fraud": "Fraud Rate"}),
            use_container_width=True,
        )

    st.subheader("Raw Data Sample")
    st.dataframe(df.head(50), use_container_width=True)

# --------------------------------------------------------------------------
# PAGE 3 — Model Performance
# --------------------------------------------------------------------------
else:
    st.title("Model Performance")
    st.write("Comparison of the two classifiers trained on this dataset.")

    results_df = pd.DataFrame(metadata["results"]).T.reset_index()
    results_df.columns = ["Model", "Accuracy", "Precision", "Recall", "F1-Score"]
    st.dataframe(results_df, use_container_width=True, hide_index=True)

    st.plotly_chart(
        px.bar(
            results_df.melt(id_vars="Model", var_name="Metric", value_name="Score"),
            x="Metric", y="Score", color="Model", barmode="group", range_y=[0, 1],
        ),
        use_container_width=True,
    )

    st.warning(
        "**A note on these results:** this dataset's fraud label has almost no "
        "statistical correlation with the available features (age, amount, hour, "
        "location, device, etc. — all near-zero correlation with fraud, and the "
        "fraud rate is ~1% across every category). This is a common property of "
        "small synthetic datasets, and it means **no model can learn a strong "
        "signal from this particular data**, however well-tuned. Random Forest "
        "simply learns to always predict 'not fraud' (98% accuracy but 0 recall), "
        "while Logistic Regression trades accuracy for some recall by weighting "
        "the minority class. For a stronger real-world result, a dataset where "
        "fraud correlates with transaction patterns (e.g. PaySim) would be a "
        "better fit — this project is structured so swapping in such a dataset "
        "only requires replacing data/fraud.csv and re-running train_model.py."
    )
