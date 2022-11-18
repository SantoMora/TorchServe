from torchvision import models

model = models.resnet18(pretrained=True)
model.eval()

for param in model.parameters():
    param.requires_grad = False