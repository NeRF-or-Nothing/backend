FROM colmap/colmap:latest

WORKDIR /app

# Update GPG Key for Nvidia manually
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

# Install dependencies and Python 3.10
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update -y && \
    apt-get install -y \
        libssl-dev \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update -y \
    && apt-get install -y \
        python3.10 \
        python3-pip

# Install pip for Python 3.10
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN python3.10 -m pip install --no-cache-dir --upgrade -r requirements.txt

# Copy the rest of the application
COPY . .

# Command to run the application
CMD ["python3.10", "main.py", "--config=configs/default.txt"]