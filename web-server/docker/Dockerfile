FROM python:3.10.8-slim

WORKDIR /app

# Install dependencies
COPY /docker/requirements.txt .
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt

COPY . .

CMD ["python3", "main.py"]