import torch
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
import os
import random
import argparse
import random
import numpy as np
import time
import pdb
time1 = time.time()

myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


parser = argparse.ArgumentParser()
parser.add_argument("--transforms",default="origin",type=str)
parser.add_argument("--lr",default=0.01,type=float)
parser.add_argument("--random_seed",default=1212,type=int)
parser.add_argument("--epoch",default=40,type=int)
parser.add_argument("--num",default=10,type=int)
parser.add_argument("--mid_num",default=50,type=int)

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)


test_Transforms = transforms.Compose([
        # transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
class imagetransform():
    def __init__(self,args):
        super(imagetransform,self).__init__()
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        if args.transforms == 'origin':
            self.transform = self.transform
       

    def __call__(self, x):
        x1 = self.transform(x)
        return x1

train_Transforms = imagetransform(args)
#  load
train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=True,
                                             transform=train_Transforms)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=1)

test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=True,
                                            transform=test_Transforms)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=1)

myModel = torchvision.models.resnet18(pretrained=False)


inchannel = myModel.fc.in_features
if args.transforms=='dropout':
    myModel.fc = nn.Sequential(nn.Linear(inchannel, 10),nn.Dropout(p=0.1))
else:
    myModel.fc = nn.Linear(inchannel, 10)
network = myModel
network.load_state_dict(torch.load('./pretrain_model/resnet18.pth'))


# network = network1(args)
network = network.to(myDevice)

learning_rate = 0.01
myOptimzier = optim.SGD(network.parameters(), lr=learning_rate,momentum=0.9)
myLoss = torch.nn.CrossEntropyLoss()

for _epoch in range(args.epoch):
    training_loss = 0.0
    correct1 = 0
    total1 = 0
    for _step, input_data in enumerate(train_loader):
        image, label = input_data[0].to(myDevice), input_data[1].to(myDevice)  
        predict_label = network(image)

        loss = myLoss(predict_label, label)



        myOptimzier.zero_grad()
        loss.backward()
        myOptimzier.step()
        numbers, predicted1 = torch.max(predict_label.data, 1)
        total1 += label.size(0)
        correct1 += (predicted1 == label).sum().item()
        training_loss = training_loss + loss.item()
        if _step % 100 == 0:
            print('[iteration - %3d] training loss: %.3f' % (_epoch * len(train_loader) + _step, training_loss / 100))
            training_loss = 0.0
            print()
    print('Training Accuracy : %.3f %%' % (100 * correct1 / total1))
    correct = 0
    total = 0
    myModel.eval()
    for images, labels in test_loader:
        images = images.to(myDevice)
        labels = labels.to(myDevice)
        outputs = network(images)  
        numbers, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Testing Accuracy : %.3f %%' % (100 * correct / total))
   
# torch.save(network.state_dict(),'./pretrain_model/resnet18_dropout.pth')
print(time.time()-time1)
