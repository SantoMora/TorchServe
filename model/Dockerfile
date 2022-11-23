FROM pytorch/torchserve:0.3.0-cpu

ARG bucket_name
ARG model_name

ENV BUCKET_NAME=$bucket_name
ENV MODEL_NAME=$model_name

#RUN mkdir modelTrained$model_name
RUN mkdir dataset
RUN mkdir model

ADD dataset dataset
COPY index_to_name.json model/
COPY testApi.py .
COPY main.py .

CMD ["mkdir -p modelTrained/$MODEL_NAME", "&&", "mv model/index_to_name.json modelTrained$model_name/", "&&", "python", " main.py"]
