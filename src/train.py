import argparse
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import mlflow
import mlflow.lightgbm
import os
import joblib

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="Path to the data file")
    parser.add_argument("--learning_rate", type=float, default=0.1, help="Learning rate")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of estimators")
    args = parser.parse_args()

    # Enable MLflow autologging
    mlflow.lightgbm.autolog()

    # Load data
    print(f"Loading data from {args.data_path}")
    df = pd.read_csv(args.data_path)

    # Features and Target
    X = df.drop("Exited", axis=1)
    y = df["Exited"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    print("Training LightGBM model...")
    clf = lgb.LGBMClassifier(
        learning_rate=args.learning_rate,
        n_estimators=args.n_estimators,
        random_state=42
    )
    
    with mlflow.start_run():
        clf.fit(X_train, y_train)

        # Evaluate
        y_pred = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print(f"Accuracy: {acc}")
        print(f"F1 Score: {f1}")

        # Log custom metrics (autolog does many, but explicit logging is good for practice)
        mlflow.log_metric("accuracy_manual", acc)
        mlflow.log_metric("f1_score_manual", f1)
        
        # Log params (redundant with autolog but good for specific tracking)
        mlflow.log_param("data_path", args.data_path)
        
        os.makedirs("artifacts", exist_ok=True)
        joblib.dump(clf, os.path.join("artifacts", "model.pkl"))

if __name__ == "__main__":
    main()
