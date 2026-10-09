
FROM python:3.12-slim

WORKDIR /app

COPY Practical-5-ML-Continuous-Delivery/predict.py ./predict.py
COPY Practical-2-ML-CI/student_result_model.pkl ./student_result_model.pkl
COPY Practical-2-ML-CI/requirements.txt ./requirements.txt

RUN pip install --no-cache-dir -r requirements.txt

ENTRYPOINT ["python", "predict.py"]
