FROM python:3.9-slim

WORKDIR /opt/ml
ENV PYTHONUNBUFFERED=TRUE

RUN apt-get -y update && apt-get -y install --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir flask==2.3.3 gunicorn==21.2.0 scikit-learn==1.3.0 pandas==2.0.3 joblib==1.3.1 numpy==1.24.3

RUN mkdir -p /opt/ml/code /opt/ml/model /opt/ml/processing

COPY serve.py /opt/ml/code/serve.py
RUN chmod +x /opt/ml/code/serve.py

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 CMD curl -f http://localhost:8080/ping || exit 1

ENTRYPOINT ["python", "/opt/ml/code/serve.py"]
CMD ["serve"]
