import pandas as pd
import numpy as np
import os

def generate_data(n_rows=1000):
    np.random.seed(42)
    data = {
        'CreditScore': np.random.randint(300, 850, n_rows),
        'Age': np.random.randint(18, 90, n_rows),
        'Tenure': np.random.randint(0, 10, n_rows),
        'Balance': np.random.uniform(0, 250000, n_rows),
        'NumOfProducts': np.random.randint(1, 4, n_rows),
        'HasCrCard': np.random.randint(0, 2, n_rows),
        'IsActiveMember': np.random.randint(0, 2, n_rows),
        'EstimatedSalary': np.random.uniform(10000, 200000, n_rows),
        'Exited': np.random.randint(0, 2, n_rows) # Target
    }
    df = pd.DataFrame(data)
    
    output_path = os.path.join('data', 'churn.csv')
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data generated at {output_path}")

if __name__ == "__main__":
    generate_data()
