import os
import json
import requests
import pandas as pd
import numpy as np
import streamlit as st

@st.cache_resource
def load_local_model():
    path = os.path.join("artifacts", "model.pkl")
    if not os.path.exists(path):
        return None
    import joblib
    return joblib.load(path)

def get_model():
    if "local_model" in st.session_state:
        return st.session_state["local_model"]
    with st.spinner("Loading model..."):
        m = load_local_model()
    st.session_state["local_model"] = m
    return m

def predict_local(model, payload):
    df = pd.DataFrame([payload])
    proba = model.predict_proba(df)[0][1]
    pred = int(proba >= 0.5)
    return pred, float(proba)

def predict_batch_local(model, df):
    proba = model.predict_proba(df)[:, 1]
    pred = (proba >= 0.5).astype(int)
    out = df.copy()
    out["prediction"] = pred
    out["probability"] = proba
    return out

def get_endpoint():
    if "endpoint_url" in st.session_state and "endpoint_key" in st.session_state:
        return st.session_state["endpoint_url"], st.session_state["endpoint_key"]
    from dotenv import load_dotenv
    load_dotenv()
    url = os.getenv("ENDPOINT_URL", "")
    key = os.getenv("ENDPOINT_KEY", "")
    st.session_state["endpoint_url"] = url
    st.session_state["endpoint_key"] = key
    return url, key

def predict_azure(payloads):
    url, key = get_endpoint()
    if not url or not key:
        return {"error": "Missing ENDPOINT_URL or ENDPOINT_KEY"}
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    body = {"data": payloads}
    try:
        r = requests.post(url, headers=headers, data=json.dumps(body), timeout=30)
        if r.status_code != 200:
            return {"error": f"HTTP {r.status_code}: {r.text}"}
        return r.json()
    except Exception as e:
        return {"error": str(e)}

st.title("Customer Churn Prediction Dashboard")
st.caption("Test the model locally or through an Azure ML Managed Endpoint")

mode = st.radio("Mode", ["Local model", "Azure endpoint"])

st.subheader("Single Prediction")
with st.form("single_form"):
    c1, c2, c3 = st.columns(3)
    with c1:
        CreditScore = st.number_input("CreditScore", min_value=300, max_value=850, value=650)
        Age = st.number_input("Age", min_value=18, max_value=90, value=35)
        Tenure = st.number_input("Tenure", min_value=0, max_value=10, value=5)
    with c2:
        Balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=50000.0, step=100.0)
        NumOfProducts = st.number_input("NumOfProducts", min_value=1, max_value=4, value=2)
        HasCrCard = st.selectbox("HasCrCard", [0, 1], index=1)
    with c3:
        IsActiveMember = st.selectbox("IsActiveMember", [0, 1], index=1)
        EstimatedSalary = st.number_input("EstimatedSalary", min_value=10000.0, max_value=200000.0, value=60000.0, step=100.0)
    submitted = st.form_submit_button("Predict")

if submitted:
    payload = {
        "CreditScore": CreditScore,
        "Age": Age,
        "Tenure": Tenure,
        "Balance": Balance,
        "NumOfProducts": NumOfProducts,
        "HasCrCard": HasCrCard,
        "IsActiveMember": IsActiveMember,
        "EstimatedSalary": EstimatedSalary,
    }
    if mode == "Local model":
        model = get_model()
        if model is None:
            st.error("Model file not found. Run training to create artifacts/model.pkl.")
        else:
            pred, proba = predict_local(model, payload)
            label = "Churn" if pred == 1 else "No Churn"
            st.metric("Prediction", label, delta=f"{proba:.2%}")
    else:
        resp = predict_azure([payload])
        if isinstance(resp, dict) and "error" in resp:
            st.error(resp["error"])
        else:
            try:
                val = resp[0]
                label = "Churn" if int(val) == 1 else "No Churn"
                st.metric("Prediction", label)
            except Exception:
                st.write(resp)

st.subheader("Batch Prediction")
uploaded = st.file_uploader("Upload CSV with feature columns", type=["csv"])
if uploaded is not None:
    df = pd.read_csv(uploaded)
    expected_cols = [
        "CreditScore","Age","Tenure","Balance","NumOfProducts","HasCrCard","IsActiveMember","EstimatedSalary"
    ]
    if not all(col in df.columns for col in expected_cols):
        st.error(f"CSV must include columns: {', '.join(expected_cols)}")
    else:
        run_batch = st.button("Run Batch Prediction")
        if run_batch:
            if mode == "Local model":
                model = get_model()
                if model is None:
                    st.error("Model file not found. Run training to create artifacts/model.pkl.")
                else:
                    res = predict_batch_local(model, df[expected_cols])
                    st.dataframe(res)
                    st.download_button("Download Results", res.to_csv(index=False), file_name="predictions.csv")
            else:
                payloads = df[expected_cols].to_dict(orient="records")
                resp = predict_azure(payloads)
                if isinstance(resp, dict) and "error" in resp:
                    st.error(resp["error"])
                else:
                    try:
                        pred = pd.Series(resp, name="prediction")
                        out = df.copy()
                        out["prediction"] = pred.astype(int)
                        st.dataframe(out)
                        st.download_button("Download Results", out.to_csv(index=False), file_name="predictions.csv")
                    except Exception:
                        st.write(resp)
