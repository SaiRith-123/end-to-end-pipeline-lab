# ================================================================
# Experiment 3: End-to-End Automated Data Pipeline (ETL + Train)
# ================================================================
# Stages:
# Extract -> Validate -> Transform -> Train -> Evaluate -> Persist
# ================================================================

import os
import joblib
import pandas as pd

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


class DataPipeline:
    """A minimal but complete automated ML data pipeline."""

    def __init__(self, artifact_dir="artifacts"):
        self.artifact_dir = artifact_dir

        self.scaler = StandardScaler()

        self.model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Create artifact directory if it doesn't exist
        os.makedirs(self.artifact_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Stage 1: Extract
    # ------------------------------------------------------------
    def extract(self):
        data = load_iris(as_frame=True)
        df = data.frame

        print(
            f"[EXTRACT] loaded "
            f"{df.shape[0]} rows x {df.shape[1]} cols"
        )

        return df

    # ------------------------------------------------------------
    # Stage 2: Validate
    # ------------------------------------------------------------
    def validate(self, df):
        missing = int(df.isnull().sum().sum())

        assert missing == 0, "Data contains missing values!"
        assert "target" in df.columns, "Target column missing!"

        print(
            "[VALIDATE] no missing values, "
            "target present -> OK"
        )

        return df

    # ------------------------------------------------------------
    # Stage 3: Transform
    # ------------------------------------------------------------
    def transform(self, df):
        X = df.drop(columns=["target"])
        y = df["target"]

        X_scaled = self.scaler.fit_transform(X)

        print(
            f"[TRANSFORM] standard-scaled "
            f"{X.shape[1]} features"
        )

        return X_scaled, y

    # ------------------------------------------------------------
    # Stage 4: Train
    # ------------------------------------------------------------
    def train(self, X, y):
        X_tr, X_te, y_tr, y_te = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42,
            stratify=y
        )

        self.model.fit(X_tr, y_tr)

        print("[TRAIN] model fitted successfully")

        return X_te, y_te

    # ------------------------------------------------------------
    # Stage 5: Evaluate
    # ------------------------------------------------------------
    def evaluate(self, X_te, y_te):
        y_pred = self.model.predict(X_te)

        accuracy = accuracy_score(y_te, y_pred)

        print(
            f"[EVALUATE] test accuracy = "
            f"{accuracy:.3f}"
        )

        print("[REPORT] classification report:")
        print(
            classification_report(
                y_te,
                y_pred,
                digits=2
            )
        )

        return accuracy

    # ------------------------------------------------------------
    # Stage 6: Persist
    # ------------------------------------------------------------
    def persist(self):
        model_path = os.path.join(
            self.artifact_dir,
            "model.pkl"
        )

        scaler_path = os.path.join(
            self.artifact_dir,
            "scaler.pkl"
        )

        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)

        print(
            f"[PERSIST] saved model -> "
            f"{model_path}"
        )

        print(
            f"[PERSIST] saved scaler -> "
            f"{scaler_path}"
        )

    # ------------------------------------------------------------
    # Pipeline Orchestration
    # ------------------------------------------------------------
    def run(self):
        print("=== Automated Data Pipeline Started ===")

        # Extract
        df = self.extract()

        # Validate
        df = self.validate(df)

        # Transform
        X, y = self.transform(df)

        # Train
        X_te, y_te = self.train(X, y)

        # Evaluate
        self.evaluate(X_te, y_te)

        # Persist
        self.persist()

        print("=== Pipeline Completed Successfully ===")


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    pipeline = DataPipeline()
    pipeline.run()