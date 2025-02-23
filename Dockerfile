# 使用 Python 3.11 作為基礎映像
FROM python:3.11

# 設定工作目錄
WORKDIR /app

# 安裝系統工具（如果有需要）
RUN apt update && apt install -y htop

# 安裝 MLflow 及必要 Python 套件
RUN pip install mlflow==2.20.1 mlflow[genai] requests

# 複製 MLflow Custom Provider
COPY ./ml_mlflow_provider /app/ml_mlflow_provider

# 確保 Python 能找到 ml_mlflow_provider
WORKDIR /app
RUN pip install -e ./ml_mlflow_provider

# 設定 Python Path，讓 MLflow 能找到 Custom Provider
ENV PYTHONPATH "/app/ml_mlflow_provider:${PYTHONPATH}"

