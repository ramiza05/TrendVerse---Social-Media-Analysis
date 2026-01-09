import mlflow
import numpy as np
import pandas as pd
import os
import dagshub
import plotly.express as px

dagshub.init(repo_owner='l230915', repo_name='trendverse',mlflow =True)
mlflow.set_tracking_uri("https://dagshub.com/l230915/trendverse.mlflow")
os.environ['MLFLOW_TRACKING_USERNAME'] = 'ramiza05'
os.environ['MLFLOW_TRACKING_PASSWORD'] = '64dd878e54ef78c19d57a721e88482e732a8fe41'

# 1. Set the experiment (or it defaults to "Default")
mlflow.set_experiment("Trendverse-forecasting")

# Log both artifacts in MLflow
with mlflow.start_run() as run:
    mlflow.log_artifact("artifacts/plot_html_2026.html")
    mlflow.log_artifact("artifacts/trending_schedule_2026.csv")