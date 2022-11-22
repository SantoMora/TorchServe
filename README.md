# TorchServe

docker build --tag=devops_model_create .
docker run -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained devops_model_create