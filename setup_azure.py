from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential
from azure.ai.ml.entities import AmlCompute

def setup_infrastructure():
    credential = DefaultAzureCredential()
    
    # Try to load from config, or handle gracefully
    try:
        ml_client = MLClient.from_config(credential=credential)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Please ensure you are logged in and have config.json or env vars set.")
        return

    # Create Compute Cluster
    cpu_compute_target = "cpu-cluster"
    
    try:
        ml_client.compute.get(cpu_compute_target)
        print(f"Compute target '{cpu_compute_target}' already exists.")
    except Exception:
        print(f"Creating compute target '{cpu_compute_target}'...")
        cpu_cluster = AmlCompute(
            name=cpu_compute_target,
            type="amlcompute",
            size="STANDARD_DS3_V2",
            min_instances=0,
            max_instances=2,
            idle_time_before_scale_down=120,
        )
        ml_client.compute.begin_create_or_update(cpu_cluster).result()
        print(f"Compute target '{cpu_compute_target}' created.")

if __name__ == "__main__":
    setup_infrastructure()
