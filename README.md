# TorchServe
TorchServe Demo to demonstrate how the model serving process inside a MLOps environment can work using a little example deploying an PyTorch ML model.


## API Model 
### Create Docker Image
```bash
docker build --tag=model_serving_api . -f API/Dockerfile
```
### Create container
``` bash
docker run -d --name model_api -v /home/ubuntu/TorchServe/model/mr:/modelTrained -p 5000:5000 model_serving_api
```
## Create Model 
### Create Docker Image
``` bash
docker build --build-arg model_name="foodnet_resnet18" \
    --build-arg bucket_name=$BUCKET_NAME \
    --tag=model_serving_train . \
    -f model/Dockerfile
```
### Create container
``` bash
docker run -d --name model_train -v /home/ubuntu/TorchServe/model/mr:/home/model-server/modelTrained model_serving_train
```
## Model Serving 
### Create Docker Image
``` bash
docker build --build-arg model_name="foodnet_resnet18" \
    --build-arg bucket_name=$BUCKET_NAME \
    --tag=model_serving_model .
```
### Create Container
``` bash
docker run -d --name torchserve -p 8080:8080 model_serving_model
```
### Validate health of torchserve API
``` bash
curl http://localhost:8080/ping
```
### Execute model
``` bash
curl -X POST http://localhost:8080/predictions/foodnet -T sample/sample.jpg
```