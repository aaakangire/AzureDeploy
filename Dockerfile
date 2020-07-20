FROM ubuntu:18.04 
MAINTAINER aaakangire@gmail.com

RUN apt-get update -y
RUN apt-get install python3-pip -y

COPY . /app
WORKDIR /app

RUN pip3 install -r requirements.txt

CMD ["python3", "app.py"]
