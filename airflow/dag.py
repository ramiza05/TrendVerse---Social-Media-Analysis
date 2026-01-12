# dags/upload_dvc_outputs.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime
import boto3, os

BUCKET = os.getenv("S3_BUCKET", "trendverse-forecasting")
REGION = os.getenv("AWS_DEFAULT_REGION", "eu-north-1")

# List expected DVC outputs (extend as needed)
ARTIFACTS = [
    ("artifacts/trending_schedule_2026.csv", "trending_schedule_2026.csv", "text/csv"),
    ("artifacts/plot_html_2026.html", "plot_html_2026.html", "text/html"),
    ("artifacts/trending_schedule_prev.csv", "trending_schedule_prev.csv", "text/csv"),
    ("artifacts/plot_html_prev.html", "plot_html_prev.html", "text/html"),
]

def upload_file(local_path, s3_key, content_type):
    s3 = boto3.client("s3", region_name=REGION)
    with open(local_path, "rb") as f:
        s3.put_object(Bucket=BUCKET, Key=s3_key, Body=f.read(), ContentType=content_type)

def upload_all():
    for local_path, s3_key, content_type in ARTIFACTS:
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Missing artifact: {local_path}")
        upload_file(local_path, s3_key, content_type)
        print(f"✓ Uploaded {local_path} → s3://{BUCKET}/{s3_key}")

with DAG(
    dag_id="retrain_and_upload_dvc_outputs",
    start_date=datetime(2025, 12, 1),
    schedule_interval="@daily",
    catchup=False,
    tags=["dvc", "s3", "upload"]
) as dag:

    upload_task = PythonOperator(
        task_id="upload_all_artifacts",
        python_callable=upload_all
    )
    upload_task
