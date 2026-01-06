import json
import os
import joblib
import pandas as pd
import logging

def init():
    """
    This function is called when the container is initialized/started, typically after create/update of the deployment.
    You can write the logic here to perform init operations like caching the model in memory
    """
    global model
    # AZUREML_MODEL_DIR is an environment variable created during deployment.
    # It is the path to the model folder (./azureml-models/$MODEL_NAME/$VERSION)
    model_path = os.path.join(
        os.getenv("AZUREML_MODEL_DIR"), "model.pkl"
    )
    # Deserialize the model file back into a sklearn model
    # Note: If using MLflow model format, you might use mlflow.lightgbm.load_model instead
    # For this POC, assuming we saved it as a standard artifact or MLflow model
    # If MLflow, we might just rely on MLflow's scoring server, but custom score.py demonstrates control.
    
    # Placeholder: In a real MLflow scenario, Azure ML can auto-generate this.
    # But writing it manually demonstrates understanding.
    try:
        model = joblib.load(model_path)
    except Exception as e:
        logging.info(f"Model not found at {model_path}, using dummy for POC or handle error: {e}")
        model = None

def run(raw_data):
    """
    This function is called for every invocation of the endpoint to perform the actual scoring/prediction.
    In the example we extract the data from the json input and call the scikit-learn model's predict()
    method and return the result back
    """
    logging.info("Request received")
    data = json.loads(raw_data)["data"]
    data = pd.DataFrame(data)
    
    if model:
        result = model.predict(data)
        return result.tolist()
    else:
        return {"error": "Model not initialized"}
