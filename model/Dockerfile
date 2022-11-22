FROM pytorch/torchserve:0.3.0-cpu

RUN mkdir model
RUN mkdir dataset

ADD dataset dataset
COPY index_to_name.json /model
COPY main.py .
RUN ls -la

CMD ["python", " main.py"]
