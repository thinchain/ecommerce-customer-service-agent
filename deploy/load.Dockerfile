FROM python:3.12-slim
RUN pip install --no-cache-dir 'locust>=2.32,<3'
COPY tests/load/locustfile.py /tests/locustfile.py
ENTRYPOINT ["locust"]
