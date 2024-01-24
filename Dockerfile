# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.10-slim

EXPOSE 5002

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements
WORKDIR /app

RUN apt-get update \
 && apt-get install --assume-yes --no-install-recommends --quiet \
        python3 \
        python3-pip \
 && apt-get -y install gcc python3-dev && apt-get -y install build-essential && apt-get install ffmpeg libsm6 libxext6  -y && apt-get clean all

RUN pip install --no-cache --upgrade pip setuptools
COPY requirements.txt .
RUN python -m pip install -r requirements.txt

COPY . /app

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
CMD [ "python3", "-m" , "flask", "run", "--host", "0.0.0.0", "--port", "5002"]