FROM python:3.12-slim

WORKDIR /workspace
COPY pyproject.toml README.md ./
COPY src ./src
COPY ui ./ui
COPY configs ./configs
RUN python -m pip install --no-cache-dir ".[api,ui]"

EXPOSE 8000 8501
CMD ["nmt", "api", "--host", "0.0.0.0", "--port", "8000"]
