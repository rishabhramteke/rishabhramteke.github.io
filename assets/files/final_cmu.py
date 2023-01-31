
#author: Rishabh Ramteke (IIT Bombay Final Year Undergraduate)
#CMU Internship 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

path = '../content/drive/My Drive/Disney5'
print( os.listdir(path))

import os
for dirname, _, filenames in os.walk(path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

#####Import Libraries

import os
from PIL import Image
import scipy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import time
from datetime import datetime
import matplotlib.pyplot as plt
import csv
import torchvision.transforms.functional as TF


dir1 = path
vec1 = os.listdir(dir1)

dir2 = dir1 + '/' + 'training'
vec2 = os.listdir(dir2)
dir3 = dir1 + '/' + 'test'
vec3 = os.listdir(dir3)

print(dir1)
print(vec1)
print(vec2)


#####Hyperparameters
batch_size = 50
num_epochs = 30       #number of epochs, default = 15
lr = 0.001            #learning rate, default = 0.01
#validation_split = 0.1
num_classes= len(vec2)

###### Transformations
transform = transforms.Compose([transforms.Resize((224,398)), 
                                transforms.ToTensor(), 
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ])
#To downscale the image, first resize by (64,64) and then resize by (224,398)

transform_train = transforms.Compose([transforms.Resize((224,398)),
                                     transforms.RandomPerspective(), 
                                     transforms.RandomRotation(45, resample=False,expand=False, center=None), 
                                     #transforms.RandomHorizontalFlip(p=0.5),
                                     transforms.ColorJitter(brightness=0.7, contrast=0.5, saturation=0.2, hue=0),
                                     transforms.RandomAffine(30, translate=None, scale=None,shear=None, resample=False, fillcolor=0),
                                     transforms.ToTensor(), 
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) , ])


trainset = torchvision.datasets.ImageFolder(dir2, transform=transform_train)
testset = torchvision.datasets.ImageFolder(dir3, transform=transform)
print(len(trainset))
print(len(testset))

data_validate, randomdata = torch.utils.data.random_split(testset, [200,len(testset)-200] )

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size ,shuffle = True)
valloader = torch.utils.data.DataLoader(data_validate, batch_size=len(data_validate) ,shuffle = True) 

for images, labels in trainloader:
    print("Image batch dimensions:", images.shape)
    print("Image label dimensions:", labels.shape)
    break

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### Pretrain model
premodel = torch.hub.load('pytorch/vision:v0.6.0', 'mobilenet_v2', pretrained=True)
premodel.to(device)

### Define model network
class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=premodel.classifier[1].in_features, out_channels=32, kernel_size=5, stride=1, padding=1, dilation=1)
        self.conv2 = nn.Conv2d(32, 64, 3, stride=1, padding=2, dilation=1)
        self.dropout1 = nn.Dropout(p=0.3, inplace=False)
        self.fc1 = nn.Linear(64*2, num_classes) 
      
    def forward(self, X):
        x = premodel.features(X)
        
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 3, 3)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = self.dropout1(x)
        x = x.view(-1, 64*2)
        x = self.fc1(x)
        out = F.log_softmax(x, dim=1)
        
        return out


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        #print("datatype : ", data.type())
        optimizer.zero_grad()
        # print(data.shape)
        # print(target.shape)
        output = model(data)
        # print(output.shape)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(target.view_as(pred)).sum().item()
    return 100. * correct / len(train_loader.dataset)

def test(model, device, test_loader,epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # access Variable's tensor, copy back to CPU, convert to numpy
            arry = output.data.cpu().numpy()
            # write CSV
            np.savetxt('output.csv', arry)

            test_loss += criterion(output, target)# sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            arry = pred.data.cpu().numpy()
            np.savetxt('pred.csv', arry)
            arry = target.data.cpu().numpy()
            np.savetxt('target.csv', arry)
            correct += pred.eq(target.view_as(pred)).sum().item()
            
            
            if(epoch==num_epochs):
            # Store wrongly predicted images of the last training epoch
              wrong_idx = (pred != target.view_as(pred)).nonzero()[:, 0]
              wrong_idx = np.array(wrong_idx.cpu())
              print(wrong_idx)
              print(len(wrong_idx))
              wrong_samples = data[wrong_idx]
              wrong_preds = pred[wrong_idx]
              actual_preds = target.view_as(pred)[wrong_idx]

              for i in range(len(wrong_idx)):
                  
                  sample = wrong_samples[i].cpu()
                  wrong_pred = wrong_preds[i]
                  actual_pred = actual_preds[i]
                  # Undo normalization
                  sample = sample * 0.225
                  sample = sample + 0.456
                  sample = sample * 255.
                  sample = sample.byte()
                  img = TF.to_pil_image(sample)
                  plt.imshow(img)
                  #img.show('wrong_idx{}_pred{}_actual{}.png'.format(wrong_idx[i], wrong_pred.item(), actual_pred.item()))
                
    test_loss /= len(test_loader.dataset)
    return 100. * correct / len(test_loader.dataset)


model = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optimizer = optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)


print("device: ",device)    
print ("optimizer chosen")
print("#######__parameters__######")
print("learning rate: ", lr, "\nepochs: ", num_epochs)
print("############################")    
print("model:\n",model)
print("############################")
print("optimizer:\n",optimizer)
print("############################")

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()

for epoch in range(1, num_epochs + 1):
    train_acc = train(model, device, trainloader, optimizer, epoch)
    val_acc = test(model, device, valloader, epoch)
    print("Epoch number: ", epoch)
    print("Train accuracy: ", train_acc)
    print("test accuracy: ", val_acc)

print("Done training.")
end.record()

Waits for everything to finish running
torch.cuda.synchronize()
print(start.elapsed_time(end))

##### Save the model
#save_path = '../content/drive/My Drive/5model.pt'
#torch.save(model, save_path)
