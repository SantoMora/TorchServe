FROM pytorch/torchserve:0.3.0-cpu


COPY ./model model/
COPY ./deployment deployment/

RUN torch-model-archiver --model-name foodnet_resnet18 \
    --version 1.0 \
    --model-file model/model.py \
    --serialized-file model/foodnet_resnet18.pth \
    --handler model/handler.py \
    --extra-files model/index_to_name.json

CMD ["torchserve", \
     "--start", \
     "--ncs", \
     "--ts-config deployment/config.properties", \
     "--model-store deployment/model-store", \
     "--models foodnet=foodnet_resnet18.mar"]
