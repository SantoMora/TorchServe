FROM pytorch/torchserve:0.3.0-cpu

RUN mkdir modelTrained
RUN mkdir dataset

ADD dataset dataset
COPY index_to_name.json /model
COPY main.py .
RUN ls -la

CMD ["python", " main.py", "&&", "torch-model-archiver", "--model-name", "foodnet_resnet18", \
    "--version 1.0", \
    "--model-file", "model/model.py", \
    "--serialized-file", "model/foodnet_resnet18.pth", \
    "--handler", "model/handler.py", \
    "--extra-files", "model/index_to_name.json"]
