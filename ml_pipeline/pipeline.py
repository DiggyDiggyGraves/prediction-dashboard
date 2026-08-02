import os
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib

class MLPipeline:
    def __init__(self, model_save_path="ml_pipeline/models/model.pkl"):
        self.model_save_path = model_save_path
        self.scaler = StandardScaler()
        self.model = LogisticRegression()

    def ingest_data(self):
        print("[*] Step 1: Ingesting data...")
        X, y = make_classification(n_samples=1000, n_features=20, n_classes=2, random_state=42)
        print(f"    Loaded dataset with shape: X={X.shape}, y={y.shape}")
        return X, y

    def preprocess_data(self, X, y):
        print("[*] Step 2: Preprocessing and splitting data...")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("    Data scaled and split successfully.")
        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_model(self, X_train, y_train):
        print("[*] Step 3: Training model...")
        self.model.fit(X_train, y_train)
        print("    Training complete.")

    def evaluate_model(self, X_test, y_test):
        print("[*] Step 4: Evaluating model...")
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        print(f"    Model Accuracy: {acc * 100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, predictions))
        return acc

    def save_artifacts(self):
        print(f"[*] Step 5: Saving model and scaler to {self.model_save_path}...")
        os.makedirs(os.path.dirname(self.model_save_path), exist_ok=True)
        joblib.dump({'model': self.model, 'scaler': self.scaler}, self.model_save_path)
        print("    Artifacts saved successfully.")

    def run(self):
        print("=== STARTING ML PIPELINE ===")
        X, y = self.ingest_data()
        X_train, X_test, y_train, y_test = self.preprocess_data(X, y)
        self.train_model(X_train, y_train)
        self.evaluate_model(X_test, y_test)
        self.save_artifacts()
        print("=== PIPELINE FINISHED SUCCESSFULLY ===")

if __name__ == '__main__':
    pipeline = MLPipeline()
    pipeline.run()
