FROM pytorch/torchserve:0.3.0-cpu
#ubuntu:20.04

#RUN apt update
#RUN apt install --no-install-recommends -y openjdk-11-jre-headless
#RUN apt install python3 -y
#RUN apt install python3-pip -y
RUN apt install curl -y
RUN apt-get install unzip

#RUN pip3 install torch==1.7.0 torchvision==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip3 install torchserve==0.2.0 torch-model-archiver==0.2.0
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" 
RUN unzip awscliv2.zip 
RUN ./aws/install
RUN pip3 install boto3

COPY credentials.csv .
COPY main.py .
COPY hola.txt .
RUN mkdir model
COPY botoTest.py .
RUN aws configure import --csv file://credentials.csv

CMD ["python3", "botoTest.py"]
