from copy import deepcopy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models.resnet import ResNet, BasicBlock
import numpy as np
import json
import os

DATA_PATH = './dataset'
BATCH_SIZE = 16
EPOCHS = 15

class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)

        self.fc = nn.Sequential(
            nn.Linear(512 * BasicBlock.expansion, 128),
            nn.ReLU(),
            nn.Dropout(.2),
            nn.Linear(128, 10),
            nn.LogSoftmax(dim=1)
        )

def setClasses(train_data_path, test_data_path):
    train_classes = dict()
    test_classes = dict()

    for path in sorted(os.listdir(train_data_path)):
        if os.path.isdir(os.path.join(train_data_path, path)):
            train_classes.setdefault(len(train_classes), path)

    with open('./model/index_to_name.json', 'w') as fp:
        json.dump(train_classes, fp)

    for path in sorted(os.listdir(test_data_path)):
        if os.path.isdir(os.path.join(test_data_path, path)):
            test_classes.setdefault(len(test_classes), path)
            
    return train_classes, test_classes

def setTransformations():
    train_transform = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_transform = T.Compose([
        T.Resize((256,256)),
        T.CenterCrop((224,224)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform, test_transform

def setDatasets(train_data_path, test_data_path, train_transform, val_transform):
    train_dataset = ImageFolder(
        root=train_data_path,
        transform=train_transform
    )

    val_dataset = ImageFolder(
        root=train_data_path,
        transform=val_transform
    )

    test_dataset = ImageFolder(
        root=test_data_path,
        transform=val_transform
    )

    return train_dataset, val_dataset, test_dataset

def defDatasetSplit(train_dataset):
    val_size = .2
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(val_size * num_train))
    random_seed = 42 
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_idx, val_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler = SubsetRandomSampler(val_idx)
    return train_sampler, val_sampler

def setDataLoaders(train_dataset, train_sampler, val_dataset, val_sampler, test_dataset):
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=train_sampler
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        num_workers=0,
        shuffle=False
    )

    return train_loader, val_loader, test_loader

def setSequentialLayer(model):
    n_inputs = model.fc.in_features
    n_outputs = 10
    sequential_layers = nn.Sequential(
        nn.Linear(n_inputs, 128),
        nn.ReLU(),
        nn.Dropout(.2),
        nn.Linear(128, n_outputs),
        nn.LogSoftmax(dim=1)
    )
    return sequential_layers

def setOpimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=1e-3, momentum=.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=.1)
    return criterion, optimizer, scheduler

def trainModel(model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset):
    loaders = {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }
    dataset_sizes = {
        'train': len(train_dataset),
        'val': len(val_dataset),
        'test': len(test_dataset)
    }

    for epoch in range(1, EPOCHS+1):
        best_acc = .0
        print(f"\nEpoch {epoch}/{EPOCHS}\n{'='*25}")
        for phase in ['train', 'val']:
            running_loss = .0
            running_corrects = .0
            if phase == 'train': model.train()
            if phase == 'val': model.eval()
            for inputs, labels in loaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels)
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            if phase == 'train': scheduler.step()
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = deepcopy(model.state_dict())
            print(f"Loss ({phase}): {epoch_loss}, Acc ({phase}): {epoch_acc}")
    torch.save(best_model_weights, './model/foodnet_resnet18.pth')
    print(f"Test Loss: {epoch_loss}, Test Accuracy: {epoch_acc}")
    model.eval()

def defineModelArchitecture(): 
    mod = ImageClassifier()
    mod.load_state_dict(torch.load("./model/foodnet_resnet18.pth"))
    mod.eval()

if __name__ == '__main__':
    train_data_path = os.path.join(DATA_PATH, 'train')
    test_data_path = os.path.join(DATA_PATH, 'test')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_classes, test_classes = setClasses(train_data_path, test_data_path)
    train_transform, val_transform, test_transform = setTransformations()
    train_dataset, val_dataset, test_dataset = setDatasets(train_data_path, test_data_path, train_transform, val_transform)
    train_sampler, val_sampler = defDatasetSplit(train_dataset)
    train_loader, val_loader, test_loader = setDataLoaders(train_dataset, train_sampler, val_dataset, val_sampler, test_dataset)

    model = models.resnet18(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    
    sequential_layers = setSequentialLayer(model)
    model.fc = sequential_layers
    criterion, optimizer, scheduler = setOpimizer(model)
    trainModel(model, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset)
    defineModelArchitecture()