FROM python:3.10-slim

RUN pip install --no-cache-dir poetry

WORKDIR /app

RUN adduser --disabled-password appuser

COPY poetry.lock pyproject.toml ./

RUN poetry config virtualenvs.create false \
    && poetry install --no-root --no-interaction --no-ansi

COPY . .

EXPOSE 8081

CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]