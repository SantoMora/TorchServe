FROM pytorch/torchserve:0.3.0-cpu

ARG model_name
ARG bucket_name

ENV BUCKET_NAME=$bucket_name
ENV MODEL_NAME=$model_name

COPY ./model model/
COPY ./deployment deployment/
COPY pullMar.py .

RUN mkdir model-store

CMD ["python", "pullMar.py", "&&" \
    "torchserve", \
     "--start", \
     "--ncs", \
     "--ts-config deployment/config.properties", \
     "--model-store model-store", \
     "--models foodnet=$MODEL_NAME.mar"]
