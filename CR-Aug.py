import torch
from torch.utils.tensorboard.summary import image
import torchvision
import torch.nn.functional as F
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.nn import Softmax,CrossEntropyLoss,BatchNorm2d,Dropout
import numpy as np
from gaussian_blur import GaussianBlur
import argparse
import random
import time
t1 = time.time()
myDevice = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--transforms",default="origin",type=str,choices=["randomresizecroping","origin","weight_decay","flip","colorjitter","Guassian","R_dropout","AUG-REG","combine","resize+flip"])
parser.add_argument("--lr",default=0.01,type=float)
parser.add_argument("--random_seed",default=1212,type=int)
parser.add_argument("--epoch",default=60,type=int)
parser.add_argument("--kl",default=2,type=int)
parser.add_argument("--save",default=False,choices=['True','False'])
parser.add_argument("--alpha",default=0.5,type=float,choices=[0.1,0.2,0.5,0.6,0.8,1.0])

args = parser.parse_args()
random.seed(args.random_seed)
np.random.seed(args.random_seed)
torch.manual_seed(args.random_seed)
myWriter = SummaryWriter('./tensorboard/log1/')


imagenet_mean_std = [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]
class imagetransform():
    def __init__(self,args,mean_std=imagenet_mean_std):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4,0.4,0.4,0.1)],p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([GaussianBlur(kernel_size=224 // 20 * 2 + 1, sigma=(0.1,2.0))],p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform1 = transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.2,1.0)),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform2 = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform3 = transforms.Compose([
            transforms.RandomApply([transforms.ColorJitter(0.2,0.2,0.2,0.05)],p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform4 = transforms.Compose([
            transforms.RandomApply([GaussianBlur(kernel_size=1, sigma=(0.05,1.0))],p=0.25),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform5 = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform6 = transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        self.transform7 = transforms.Compose([
            transforms.RandomResizedCrop(224,scale=(0.2,1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(*mean_std)
        ])
        if args.transforms == 'randomresizecroping':
            self.transform = self.transform1
        elif args.transforms == 'flip':
            self.transform = self.transform2
        elif args.transforms == 'colorjitter':
            self.transform = self.transform3
        elif args.transforms == 'Guassian':
            self.transform = self.transform4
        elif args.transforms == 'R_dropout' or args.transforms == 'weight_decay' or args.transforms == 'origin':
            self.transform = self.transform5
        elif args.transforms == 'combine':
            self.transform = self.transform6
        elif args.transforms == 'resize+flip':
            self.transform = self.transform7
        else:
            self.transform =self.transform

    def __call__(self, x):
        x1 = self.transform(x)
        x2 = self.transform(x)
        return x1, x2
augmentation = imagetransform(args)
if args.transforms=='randomresizecroping' or args.transforms=='combine' or args.transforms=='resize+flip' or args.transforms=='AUG-REG':
    test_Transform = transforms.Compose([
        transforms.Resize([224,224]),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
else:
    test_Transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

train_dataset = torchvision.datasets.CIFAR10(root='./', train=True, download=False,
                                              transform=augmentation) 
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)

test_dataset = torchvision.datasets.CIFAR10(root='./', train=False, download=False,
                                            transform=test_Transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True, num_workers=4)


network = torchvision.models.resnet18(pretrained=False)


inchannel = network.fc.in_features
if args.transforms=='R_dropout':
    network.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.1,training=m.training))
    network.fc = nn.Sequential(nn.Linear(inchannel, 10))
    network.load_state_dict(torch.load('./pretrain_model/resnet18_dropout.pth'))
else:
    network.fc = nn.Linear(inchannel,10)
    # network.load_state_dict(torch.load('./pretrain_model/resnet18_aug.pth'))
# if args.transforms=='randomresizecroping' or args.transforms=='combine' or args.transforms=='resize+flip' or args.transforms=='AUG-REG':
#     network.load_state_dict(torch.load('./pretrain_model/resnet18_aug224.pth'))
# if args.transforms=='R_dropout':
#     network.load_state_dict(torch.load('./pretrain_model/resnet18_dropout.pth'))
# else:
#     network.load_state_dict(torch.load('./pretrain_model/resnet18_aug.pth'))




class net_struct(nn.Module):
    def __init__(self,args):
        super(net_struct,self).__init__()
        self.net = network
    def forward(self,x1,x2=None,y=None):
        x1 = self.net(x1)
        loss_f = CrossEntropyLoss()
        if y is not None:
            loss = loss_f(x1, y)
            x2 = self.net(x2)
            loss2 = loss_f(x2, y)
            loss += loss2  
            p1 = torch.softmax(x1,dim=1)
            p2 = torch.softmax(x2,dim=1)
            if args.kl==1:
                q1 = torch.log_softmax(x1,dim=1)
                q2 = torch.log_softmax(x2,dim=1)
                loss4 = torch.nn.functional.kl_div(q1,p2,reduction='none').sum() + torch.nn.functional.kl_div(q2,p1,reduction='none').sum()
            if args.kl==3:
                # q1 = torch.log_softmax(x1,dim=1)
                q2 = torch.log_softmax(x2,dim=1)
                loss4 = torch.nn.functional.kl_div(q2,p1,reduction='none').sum()
            else:
                loss4 = 1 - F.cosine_similarity(p1,p2.detach(),dim=-1).mean() 
            loss += args.alpha*loss4
            return loss
        else:
            return x1
# network = network.to(myDevice)
network1 = net_struct(args)
network1 = network1.to(myDevice) 
# network = nn.DataParallel(network)
# network1 = nn.DataParallel(network1)
# network.cuda()
# network1.cuda()
if args.transforms == 'weight_decay':
    optimizer = optim.SGD(network1.parameters(),lr=args.lr,momentum=0.9,weight_decay=0.1)
optimizer = optim.SGD(network1.parameters(),lr=args.lr,momentum=0.9)

train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(20 + 1)]
def train(epoch):
    all_pred1 = []
    all_target1 = []
    network1.train()
    for batch_idx, ((data1,data2), target) in enumerate(train_loader):
        # print(data1.shape,target.shape)
        data1 = data1.to(myDevice)
        data2 = data2.to(myDevice)
        target = target.to(myDevice)
        optimizer.zero_grad()
        loss = network1(data1,data2, target)

        loss = loss.mean()                  
       
        loss.backward()
        optimizer.step()
        output1 = network1(data1)
        preds = torch.argmax(output1, dim=-1)
        if batch_idx % 100 == 0:
            print('Train Epoch: {}[{}/{}({:.0f}%)]\tloss:{:.6f}'.format(
                epoch, batch_idx * len(data1), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
        if len(all_pred1) == 0:
            all_pred1.append(preds.detach().cpu().numpy())
            all_target1.append(target.detach().cpu().numpy())
        else:
            all_pred1[0] = np.append(all_pred1[0], preds.detach().cpu().numpy(), axis=0)
            all_target1[0] = np.append(all_target1[0], target.detach().cpu().numpy(), axis=0)
#     print(all_pred)
    all_pred1, all_target1 = all_pred1[0], all_target1[0]
    acc = (all_pred1 == all_target1).mean()
    print('train:',acc) 


def test(epoch):
    all_pred = []
    all_target = []
    network1.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            data = images.to(myDevice)
            target = labels.to(myDevice)
            output = network1(data)
            output = torch.log_softmax(output, dim=-1)
            test_loss = F.nll_loss(output, target)
            preds = torch.argmax(output, dim=-1)
            #             acc = (pred==target).mean()
            if len(all_pred) == 0:
                all_pred.append(preds.detach().cpu().numpy())
                all_target.append(target.detach().cpu().numpy())
            else:
                all_pred[0] = np.append(all_pred[0], preds.detach().cpu().numpy(), axis=0)
                all_target[0] = np.append(all_target[0], target.detach().cpu().numpy(), axis=0)
    #     print(all_pred)
    all_pred, all_target = all_pred[0], all_target[0]
    acc = (all_pred == all_target).mean()
    print(acc)
    with open('./log/resnet18_' + args.transforms + '_aug.txt', 'a') as f:
     print('epoch : %d'%(epoch)+ '    ' + 'Testing Accuracy : %.3f' % (acc), file=f)
for epoch in range(1, args.epoch):
    train(epoch)
    test(epoch)
if args.save:
    torch.save(network.state_dict(),'./pretrain_model/resnet18_aug' + args.transforms + '.pth')
print(time.time()-t1)

