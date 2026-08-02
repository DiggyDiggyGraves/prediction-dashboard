# Prediction Dashboard

A full-stack application featuring a machine learning pipeline backend and an interactive frontend dashboard.

## Tech Stack
* **Backend:** Python, FastAPI
* **Frontend:** React
* **Machine Learning:** Custom Python ML pipeline and neural network components

## Project Structure
* `app.py` - Main FastAPI backend server
* `frontend/` - React frontend dashboard interface
* `ml_pipeline/` - Machine learning training and prediction pipeline
* `neural_net/` - Neural network architecture and model logic

## Getting Started

### 1. Backend Setup
Navigate to the project root and start your Python environment:
```bash
source neural_net_env/bin/activate
pip install -r requirements.txt  # (or install dependencies as needed)
uvicorn app:app --reload
