import dagshub
import mlflow
from datetime import datetime

from evaluation import run_evaluation
from src.config import MODEL_NAME
from test import run_test
from train import run_train

dagshub.init(repo_owner='wjan244', repo_name='DataChallenge2026', mlflow=True)

experiment = mlflow.set_experiment("DataChallenge_2026")

if __name__ == "__main__":
    timestamp = f"{datetime.now():%Y-%m-%d_%H:%M}"
    with mlflow.start_run(experiment_id=experiment.experiment_id,run_name=f"{MODEL_NAME}_{timestamp}"):
        run_train(timestamp)
        run_evaluation(timestamp)
        run_test(timestamp)
        
    mlflow.end_run()