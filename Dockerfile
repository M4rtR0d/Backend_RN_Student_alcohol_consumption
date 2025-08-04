FROM python:3.10-slim

# Ensure system packages are up to date and security patches are applied

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN apt-get update && apt-get upgrade -y && apt-get clean && rm -rf /var/lib/apt/lists/* \
	&& pip install --no-cache-dir -r /app/requirements.txt

COPY . .

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]