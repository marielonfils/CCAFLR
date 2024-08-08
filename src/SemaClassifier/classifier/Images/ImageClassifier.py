import torch.nn as nn
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedShuffleSplit
from PIL import Image
import numpy as np
import time

colours = ['\033[32m', '\033[33m', '\033[34m', '\033[35m','\033[36m', '\033[37m', '\033[90m', '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
reset = '\033[0m'
bold = '\033[01m'
disable = '\033[02m'
underline = '\033[04m'
reverse = '\033[07m'
strikethrough = '\033[09m'
invisible = '\033[08m'
default='\033[00m'

BATCH=8
BATCH_TEST=16


def cprint(text,id):
    print(f'{colours[id%13]} {text}{default}')

#from https://raw.githubusercontent.com/developer0hye/Custom-CNN-based-Image-Classification-in-PyTorch/master/main.py
class ConvNet(nn.Module):
    def __init__(self, num_classes):
        super(ConvNet, self).__init__()

        self.layer1 = self.conv_module(3, 16)
        self.layer2 = self.conv_module(16, 32)
        self.layer3 = self.conv_module(32, 64)
        self.layer4 = self.conv_module(64, 128)
        self.layer5 = self.conv_module(128, 256)
        self.gap = self.global_avg_pool(256, num_classes)
        self.num_classes = num_classes

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.gap(out)
        out = out.view(-1, self.num_classes)

        return out

    def conv_module(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))

    def global_avg_pool(self, in_num, out_num):
        return nn.Sequential(
            nn.Conv2d(in_num, out_num, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_num),
            nn.LeakyReLU(),
            nn.AdaptiveAvgPool2d((1, 1)))


def split(path):
    labels=np.loadtxt(path+"/labels.csv",delimiter=",",dtype=str)
    sss=StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=4)
    l=len(labels)
    train,train_y, test, test_y =[],[],[],[]
    train_index, test_index=[],[]
    for tr,ts in sss.split(np.zeros(l) ,labels[:,1]):
        train_index=tr
        test_index=ts
    for i in train_index:
        p,l=labels[i]
        image, label = preprocess_image2(path+"/"+p),int(l)
        train.append(image)
        train_y.append(label)
    for i in test_index:
        p,l=labels[i]
        image, label = preprocess_image2(path+"/"+p),int(l)
        test.append(image)
        test_y.append(label)
        
    return train,train_y, test, test_y, [],[]

def preprocess_image2(img_path):
    img = Image.open(img_path).convert('RGB')
    return img
    
def train(model, dataset, batch, epochs,id):
    dataloader = DataLoader(dataset, batch_size=batch)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    size = len(dataloader.dataset)
    model.train()
    t=0
    for epoch in range(epochs):
        t1=time.time()
        l=0
        for batch, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            # Compute prediction error
            optimizer.zero_grad()
            pred = model(X)
            loss = loss_fn(pred, y)
            l+=loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()
        l=l/size
        t+=time.time()-t1
        cprint(f"Client {id}: Epoch {epoch}, Loss: {l}",id)
    cprint('--------------------FIT OK----------------',id)
    return model, {'loss':l,'train_time':t}

def test(model,dataset,batch,id):
    dataloader=DataLoader(dataset, batch_size=batch)
    loss_fn= torch.nn.CrossEntropyLoss()
    t=time.time()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    y_pred=[]
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            for i,p in enumerate(pred):
                y_pred.append(p.argmax().item())
    test_loss /= num_batches
    correct /= size
    t2=time.time()-t    
    cprint(f"Test Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n",id)
    return t2, test_loss,y_pred
    
class ImagesDataset(Dataset):
    def __init__(self,data,labels,transform=None, target_transform=None) -> None:
        super().__init__()
        self.data=data
        self.labels=labels
        self.transform = transform
        self.target_transform = target_transform
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, index):
        image,label= self.data[index],self.labels[index]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image,label