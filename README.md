# TorchServe


docker build --tag=model_serving_train .
docker build --tag=model_serving_api .

docker run -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train
docker run -v /home/ubuntu/TorchServe/model/mr:/modelTrained model_serving_api