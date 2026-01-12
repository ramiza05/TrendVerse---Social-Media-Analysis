FROM apache/airflow:2.8.4-python3.11

LABEL io.astronomer.docker=true

USER root
# Prophet needs build tools
RUN apt-get update && apt-get install -y build-essential

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dags/ /opt/airflow/dags/
