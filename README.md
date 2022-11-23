# TorchServe


docker build --build-arg model_name="foodnet_resnet18" \
    --build-arg bucket_name=$BUCKET_NAME \
    --tag=model_serving_train . \
    -f model/Dockerfile
    
docker build --tag=model_serving_api . -f API/Dockerfile

docker build --build-arg model_name="foodnet_resnet18" \
    --build-arg bucket_name=$BUCKET_NAME \
    --tag=model_serving_model .


docker run -d --name model_api -v /home/ubuntu/TorchServe/model/mr:/modelTrained -p 5000:5000 model_serving_api
docker run -it --name model_train -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train /bin/bash
docker run -d --name model_train -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train
docker run -d --name torchserve -p 8080:8080 model_serving_model

curl http://localhost:8080/ping

curl -X POST http://localhost:8080/predictions/foodnet -T sample/sample.jpg