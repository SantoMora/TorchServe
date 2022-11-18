import torch.nn as nn

sequential_layer = nn.Sequential(
    nn.Linear(model.fc.in_features, 128),
    nn.ReLU(),
    nn.Dropout(.2),
    nn.Linear(128, 10),
    nn.LogSoftmax(dim=1)
)

model.fc = sequential_layer

torch.save(model.state_dict(), '../model/foodnet_resnet18.pth')