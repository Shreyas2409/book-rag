FROM python:3.11
WORKDIR /code
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY observability.py .
COPY eval_agent.py .
COPY context_window.py .
COPY chunking.py .
COPY servers/agent_api/main.py .
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
