# Project Roadmap: Azure MLOps Job-Readiness POC
**Goal:** Build an end-to-end MLOps pipeline for Customer Churn prediction using Azure ML v2, LightGBM, and MLflow.
**Target Role:** MLOps Engineer / Machine Learning Engineer.

---

## 📂 Phase 1: Project Initialization & Environment
**Objective:** Set up the directory structure and dependency management.

- [ ] **Step 1.1: Create Directory Structure**
    - Create the following folder structure:
      ```text
      azure-mlops-poc/
      ├── .github/workflows/
      ├── data/
      ├── src/
      │   ├── __init__.py
      │   ├── train.py
      │   ├── score.py
      ├── environment/
      │   └── conda.yaml
      ├── .env.template
      ├── requirements.txt
      └── pipeline_job.py
      ```
- [ ] **Step 1.2: Define Dependencies**
    - Create `requirements.txt` containing:
      - `azure-ai-ml`
      - `azure-identity`
      - `mlflow`
      - `azureml-mlflow`
      - `lightgbm`
      - `scikit-learn`
      - `pandas`
      - `numpy`
- [ ] **Step 1.3: Create Synthetic Data Generator**
    - Create a script `data/generate_data.py` to generate a synthetic `churn.csv` (1000 rows) with features: `CreditScore`, `Age`, `Tenure`, `Balance`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, `EstimatedSalary`, and target `Exited` (0 or 1). Run it to populate `data/churn.csv`.

---

## 🛠 Phase 2: Modular Training Code (Local Development)
**Objective:** Create the training script that works locally but is ready for the cloud (MLflow integrated).

- [ ] **Step 2.1: Write `src/train.py`**
    - **Inputs:** Must accept command line arguments: `--data_path`, `--learning_rate`, `--n_estimators`.
    - **Logic:**
      1. Load data from `data_path`.
      2. Split into Train/Test.
      3. Enable `mlflow.lightgbm.autolog()`.
      4. Train a `LGBMClassifier`.
      5. Log custom metrics (Accuracy, F1-Score) to MLflow.
      6. Save the model locally as an artifact if needed.
- [ ] **Step 2.2: Test Local Training**
    - Verify the script runs via terminal: `python src/train.py --data_path ./data/churn.csv`

---

## ☁️ Phase 3: Azure ML Pipeline Definition (SDK v2)
**Objective:** Define the infrastructure and pipeline logic using Python SDK v2.

- [ ] **Step 3.1: Define Environment (`environment/conda.yaml`)**
    - Create the YAML file defining the Docker environment for Azure ML (Python 3.8+, pip dependencies from Step 1.2).
- [ ] **Step 3.2: Create `pipeline_job.py`**
    - **Logic:**
      1. Use `DefaultAzureCredential` for auth.
      2. Get a handle to the `MLClient`.
      3. Define a `command` job that:
         - Uses the code in `./src`.
         - Runs `python train.py`.
         - Mounts the data input.
         - Uses the environment defined in Step 3.1.
         - Sets the `compute` target (e.g., "cpu-cluster").
- [ ] **Step 3.3: Infrastructure Setup Script (Optional Helper)**
    - Create `setup_azure.py` to:
      - Create the MLClient.
      - Create a Compute Cluster named `cpu-cluster` if it doesn't exist.

---

## 🚀 Phase 4: Model Deployment & Serving
**Objective:** Create the artifacts required for real-time inference.

- [ ] **Step 4.1: Write `src/score.py`**
    - Implement `init()`: Load the model from `AZUREML_MODEL_DIR`.
    - Implement `run(raw_data)`: Accept JSON input, parse it into a DataFrame, predict using the loaded model, return JSON response.
- [ ] **Step 4.2: Define Endpoint Config (YAML)**
    - Create `deployment/endpoint.yaml` and `deployment/deployment.yaml` to define a **Managed Online Endpoint**.

---

## 🔄 Phase 5: CI/CD Automation (GitHub Actions)
**Objective:** Automate the pipeline trigger on code push.

- [ ] **Step 5.1: Create Workflow `.github/workflows/mlops-pipeline.yml`**
    - **Trigger:** Push to `main`.
    - **Steps:**
      1. Checkout code.
      2. Install Python dependencies.
      3. Log in to Azure (using `AZURE_CREDENTIALS` secret).
      4. Run `pipeline_job.py` to trigger the training job in Azure.

---

## ✅ Phase 6: Documentation (The "Hire Me" Factor)
**Objective:** Explain the architecture clearly.

- [ ] **Step 6.1: Update README.md**
    - Add an architecture diagram (mermaid.js).
    - Explain how to run the local training.
    - Explain how the Azure Pipeline works.
    - List the "Job Ready" skills demonstrated (MLflow, Azure SDK v2, CI/CD).