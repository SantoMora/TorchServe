FROM ubuntu:20.04

RUN apt update
RUN apt install --no-install-recommends -y openjdk-11-jre-headless
RUN apt install python3 -y
RUN apt install python3-pip -y
RUN apt install curl -y
RUN apt-get install unzip -y

RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
RUN unzip awscliv2.zip 
RUN ./aws/install

COPY credentials.csv .
COPY api.py .
COPY requirements.txt .
RUN pip3 install -r requirements.txt
RUN aws configure import --csv file://credentials.csv

EXPOSE 5000
CMD ["python3", "api.py"]
