FROM docker.io/library/python:3.11.0

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

ENV HOME=/home/docker_user
WORKDIR ${HOME}

COPY wait-for-it.sh  ${HOME}/wait-for-it.sh
RUN chmod +x wait-for-it.sh

COPY requirements.txt ${HOME}/requirements.txt
RUN pip install -r requirements.txt

COPY src/ ${HOME}/src
COPY tests/ ${HOME}/tests

RUN pip install -e ${HOME}/src