from azure.ai.ml import MLClient, command, Input
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import Environment

def main():
    # Connect to Azure ML Workspace
    # Assumes config.json is present or params are passed, or env vars are set
    # fallback to DefaultAzureCredential which works with AZ login
    credential = DefaultAzureCredential()
    
    # Replace these with your actual details if not using config.json or env vars
    # subscription_id = "..."
    # resource_group = "..."
    # workspace_name = "..."
    
    try:
        ml_client = MLClient.from_config(credential=credential)
    except Exception:
        # If config not found, assume env vars or manual input (placeholder here)
        print("config.json not found, ensure you have AZURE_SUBSCRIPTION_ID, AZURE_RESOURCE_GROUP, AZURE_WORKSPACE_NAME set.")
        return

    # Define the job
    job = command(
        code="./src",  # upload the src directory
        command="python train.py --data_path ${{inputs.churn_data}} --learning_rate ${{inputs.learning_rate}} --n_estimators ${{inputs.n_estimators}}",
        inputs={
            "churn_data": Input(
                type="uri_file",
                path="./data/churn.csv", # This will be uploaded
            ),
            "learning_rate": 0.1,
            "n_estimators": 100,
        },
        environment="azure-mlops-poc-env@latest", # Assumes env is registered, or define inline
        # Ideally, we register the environment first using conda.yaml
        # For simplicity in POC, we can use a base image + conda_file
        environment=Environment(
            image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
            conda_file="./environment/conda.yaml",
        ),
        compute="cpu-cluster", # Ensure this compute target exists
        display_name="churn-prediction-job",
        experiment_name="churn-prediction-experiment",
        description="Train a LightGBM model for Churn Prediction"
    )

    # Submit the job
    print("Submitting job...")
    returned_job = ml_client.create_or_update(job)
    print(f"Job submitted. Studio URL: {returned_job.studio_url}")

if __name__ == "__main__":
    main()
