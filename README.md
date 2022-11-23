# TorchServe


docker build --build-arg model_name="foodnet_resnet18" \
    --build-arg bucket_name=$BUCKET_NAME \
    --tag=model_serving_train .
    
docker build --tag=model_serving_api .
docker build --tag=model_serving_model .

docker run -it --name model_train -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train /bin/bash
docker run --name model_train -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train
docker run -d --name model_api -v /home/ubuntu/TorchServe/model/mr:/modelTrained -p 5000:5000 model_serving_api