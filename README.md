# Azure MLOps POC: Customer Churn Prediction

This project is a Proof of Concept (POC) demonstrating an end-to-end MLOps pipeline for Customer Churn prediction using Azure Machine Learning v2, LightGBM, and MLflow. It is designed to showcase job-readiness for MLOps/MLE roles.

## 📌 Architecture Overview

The pipeline follows a modular architecture:
1.  **Data Generation**: Synthetic data generation (`data/generate_data.py`).
2.  **Model Training**: Modular script (`src/train.py`) using LightGBM and Scikit-Learn, integrated with MLflow for experiment tracking.
3.  **Azure ML Pipeline**: Defined via Python SDK v2 (`pipeline_job.py`) to orchestrate training on Azure Compute Clusters.
4.  **Deployment**: Managed Online Endpoint configuration (`deployment/`) for real-time inference.
5.  **CI/CD**: GitHub Actions workflow (`.github/workflows/`) for automated training and deployment.

## 📂 Directory Structure

```text
azure-mlops-poc/
├── .github/workflows/   # CI/CD Pipelines
├── data/                # Data storage (local)
├── src/                 # Source code
│   ├── train.py         # Training logic
│   ├── score.py         # Inference logic
├── environment/         # Environment definitions
│   └── conda.yaml       # Conda environment for Azure
├── deployment/          # Deployment configurations
│   ├── endpoint.yaml    # Endpoint definition
│   └── deployment.yaml  # Deployment definition
├── streamlit_app.py     # Interactive dashboard (local & Azure endpoint)
├── .env.template        # Environment variables for Azure endpoint
├── pipeline_job.py      # Azure ML Pipeline definition
├── setup_azure.py       # Helper to setup infrastructure
└── requirements.txt     # Local dependencies
```

## 🚀 How to Run

### 1. Local Development (Testing the Code)

Ensure you have Python 3.8+ installed.

```bash
# Install dependencies
pip install -r requirements.txt

# Generate synthetic data
python data/generate_data.py

# Run training locally to verify logic
python src/train.py --data_path data/churn.csv
```

### 2. Azure ML Pipeline (Cloud Execution)

Ensure you have the Azure CLI installed and are logged in (`az login`).

```bash
# Setup Infrastructure (Compute Cluster)
python setup_azure.py

# Submit the Training Job
python pipeline_job.py
```

### 3. Deployment

Deploy the trained model as a real-time endpoint:

```bash
az ml online-endpoint create --file deployment/endpoint.yaml
az ml online-deployment create --file deployment/deployment.yaml --all-traffic
```

### 4. Streamlit Dashboard (Interactive Testing)

Run a user-friendly UI to test single and batch predictions locally or against an Azure endpoint.

```bash
# Ensure dependencies are installed
pip install -r requirements.txt

# Train locally to create artifacts/model.pkl
python data/generate_data.py
python src/train.py --data_path data/churn.csv

# Launch the dashboard
python -m streamlit run streamlit_app.py
```

- Open the app at http://localhost:8501
- File: [streamlit_app.py](file:///c:/Users/Asad/Desktop/Personal%20Projects%20Q4%202025/BCG%20POC/streamlit_app.py)

Hosted demo:
- https://asad-telco.streamlit.app/

To use the Azure endpoint mode, create a `.env` file using:

- Template: [.env.template](file:///c:/Users/Asad/Desktop/Personal%20Projects%20Q4%202025/BCG%20POC/.env.template)
- Variables:
  - ENDPOINT_URL: Managed Online Endpoint invoke URL
  - ENDPOINT_KEY: Endpoint key

Then select “Azure endpoint” in the app to send requests to the deployed model.

The dashboard uses lazy loading to keep startup fast:
- The local model (artifacts/model.pkl) loads only on first prediction/batch run and is cached.
- Endpoint credentials load only when Azure mode is used and are cached.

## 🎯 Job Readiness Demonstration

This POC demonstrates the following skills required by the Job Description:

-   **Model Maintenance & Automation**: Automated pipelines via `pipeline_job.py` and GitHub Actions.
-   **Azure ML Integration**: Deep usage of Azure ML SDK v2 for Compute, Environments, and Jobs.
-   **Reusable Components**: Modular `train.py` and environment definitions.
-   **Monitoring & Logging**: MLflow integration for tracking metrics (Accuracy, F1) and parameters.
-   **Model Serving**: Managed Online Endpoints (API) with custom `score.py`.
-   **CI/CD**: GitHub Actions workflow for continuous integration.

## 🛠 Tech Stack
-   **Language**: Python
-   **ML Frameworks**: LightGBM, Scikit-Learn
-   **MLOps Platform**: Azure Machine Learning (SDK v2)
-   **Tracking**: MLflow
-   **CI/CD**: GitHub Actions
