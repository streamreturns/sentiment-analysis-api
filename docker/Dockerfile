FROM continuumio/anaconda3
MAINTAINER byeongho.kang@yahoo.com

WORKDIR /
COPY filesystem.tar.bz2 .
RUN tar xjf filesystem.tar.bz2

WORKDIR /root

RUN apt update
RUN apt upgrade -y
RUN apt install
RUN apt install -y openssh-server net-tools htop vim nano screen
RUN service ssh start

RUN git clone https://github.com/streamreturns/sentiment-analysis-api.git
RUN pip install -r sentiment-analysis-api/requirements.txt

RUN echo "FASTAPI KR" > /root/sentiment-analysis-api/config/stage_identifier

WORKDIR /root/sentiment-analysis-api
COPY koelectra-base-v3-sentiment-model.tar.bz2 .
RUN tar xjf koelectra-base-v3-sentiment-model.tar.bz2

EXPOSE 22 65280-65407