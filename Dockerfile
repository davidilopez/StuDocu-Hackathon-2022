# syntax=docker/dockerfile:1
FROM python:3.8.10-slim
COPY API/ .
COPY Pipfile .
COPY Pipfile.lock .
RUN pip install pipenv
RUN pipenv install
RUN pipenv run python -m spacy download en_core_web_sm
RUN pipenv run uvicorn app:app --reload
EXPOSE 8000