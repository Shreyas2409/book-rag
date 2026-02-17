FROM python:3.11
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY observability.py .
COPY servers/mcp_vectordb/main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
