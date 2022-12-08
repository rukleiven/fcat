# This is a docker file that will run flake8 checks and pytest
FROM python:3.8-slim

COPY . /app

RUN pip install /app flake8 pytest
CMD flake8 --append-config=/app/.flake8 /app/fcat && pytest /app/tests/
