import numpy as np
import pandas as pd
from glob import glob
import cv2
import fnmatch
from sklearn import model_selection
from sklearn.model_selection import train_test_split, KFold, cross_val_score, StratifiedKFold, learning_curve, GridSearchCV
from sklearn.metrics import confusion_matrix, make_scorer, accuracy_score
from torch.utils.data import Dataset, DataLoader
from torchvision.models.mobilenet import mobilenet_v2
import torch
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from torch.nn import CrossEntropyLoss
import torch.nn as nn
import time

class MobileNet(nn.Module):
    def __init__(self, learning_rate,reduce_lr_gamma,num_classes=2):
        super(MobileNet, self).__init__()
        #self.layers = mobilenet_v2(pretrained=True,)
        self.layers=mobilenet_v2()
        self.layers.load_state_dict(torch.load("./SemaClassifier/classifier/Breast/mobilenet.pt", weights_only=True))
        self.layers.eval()

        self.layers.classifier[1] = torch.nn.Linear(in_features=self.layers.classifier[1].in_features, out_features=num_classes)
        self.optimizer = optim.Adadelta(self.layers.parameters(), lr=learning_rate)
        self.scheduler = StepLR(self.optimizer, step_size=1, gamma=reduce_lr_gamma)
        self.num_classes=num_classes
    def forward(self, x):
        self.layers.double()
        return self.layers(x)#.view(-1, self.num_classes)


def train(model, train_dataset,epochs,batch_size,id,*args, **kwargs):
    train_loader = DataLoader(train_dataset, batch_size=32)#, shuffle=True)

    t1=time.time()
    loss_func = CrossEntropyLoss()
    model.train()
    t=time.time()-t1
    for epoch in range(epochs):
        t2=time.time()
        for batch_idx, (data,target) in enumerate(train_loader):
            model.optimizer.zero_grad()
            output=model(data)
            loss = loss_func(output, target)
            loss.backward()
            model.optimizer.step()
            #if batch_idx % log_interval == 0:
            #    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            #        epoch, batch_idx * len(data), len(train_loader.dataset),
            #        100. * batch_idx / len(train_loader), loss.item()))
        model.scheduler.step()
        t+=time.time()-t2

        cprint(f"Client {id}: Epoch {epoch}, Loss: {loss}",id)
    cprint('--------------------FIT OK----------------',id)
    return model, {'loss':loss.item(),"train_time":t}

def test(model, test_dataset,batch_size, id,*args, **kwargs):
    test_loader = DataLoader(test_dataset, batch_size=32)

    t0=time.time()
    model.eval()
    test_loss = 0
    loss_func = CrossEntropyLoss()
    t=time.time()-t0
    y_pred=[]
    with torch.no_grad():
        for data, target in test_loader:
            t1=time.time()
            output = model(data)
            test_loss += loss_func(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            t+=time.time()-t1
            for p in pred:
                y_pred.append(p.item())

    test_loss /= len(test_loader)
    cprint('--------------------TEST OK----------------',id)
    return t, test_loss, y_pred

def get_model():
    batch_size = 1000
    learning_rate = 1.0
    reduce_lr_gamma = 0.7
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    kwargs = {'batch_size': batch_size}
    if torch.cuda.is_available():
        kwargs.update({'num_workers': 1, 'pin_memory': True})

    model = MobileNet(learning_rate,reduce_lr_gamma )
    model.double()
    model.to(device)
    return model.layers

def proc_images(lowerIndex,upperIndex,imagePatches,classZero,classOne):
    """
    Returns two arrays: 
        x is an array of resized images
        y is an array of labels
    """ 
    x = []
    y = []
    WIDTH = 50
    HEIGHT = 50
    for img in imagePatches[lowerIndex:upperIndex]:
        full_size_image = cv2.imread(img)
        x.append(cv2.resize(full_size_image, (WIDTH,HEIGHT), interpolation=cv2.INTER_CUBIC))
        if img in classZero:
            y.append(0)
        elif img in classOne:
            y.append(1)
        else:
            return
    return x,y

def init_datasets_breast(nclients,id, datapath, split):
    path="./databases/Breast/archive/IDC_regular_ps50_idx5/**/*.png"
    if datapath is not None:
        path=datapath
    imagePatches = glob(path, recursive=True)
    patternZero = '*class0.png'
    patternOne = '*class1.png'
    classZero = fnmatch.filter(imagePatches, patternZero)
    classOne = fnmatch.filter(imagePatches, patternOne)
    if split: #fragment of dataset
        if id == nclients: #server
            li=id*1000
            ui=li+18000
        else:
            li=id*1000
            ui=(id+1)*1000
    else: #whole dataset
        li=0
        ui=20000
    X,Y = proc_images(li,ui,imagePatches,classZero,classOne)
    X2=np.array(X)
    X3=X2/255.0

    X_train, X_test, Y_train, Y_test = train_test_split(X3, Y, test_size=0.3)
    # Reduce Sample Size for DeBugging
    X_train2 = X_train[0:300000] 
    Y_train2 = Y_train[0:300000]
    X_test2 = X_test[0:300000] 
    Y_test2 = Y_test[0:300000]

    X_trainShape = X_train2.shape[1]*X_train2.shape[2]*X_train2.shape[3]
    X_testShape = X_test2.shape[1]*X_test2.shape[2]*X_test2.shape[3]
    X_trainFlat = X_train2.reshape(X_train2.shape[0], X_trainShape)
    X_testFlat = X_test2.reshape(X_test2.shape[0], X_testShape)
    for i in range(len(X_trainFlat)):
        height, width, channels = 50,50,3
        X_trainFlat2 = X_trainFlat.reshape(len(X_trainFlat),channels,height,width)
    for i in range(len(X_testFlat)):
        height, width, channels = 50,50,3
        X_testFlat2 = X_testFlat.reshape(len(X_testFlat),channels,height,width)
    
    train_dataset = BreastDataset(X_trainFlat2,Y_train2)
    test_dataset = BreastDataset(X_testFlat2,Y_test2)
    return train_dataset, Y_train2, test_dataset, Y_test2, ['IDC','non-IDC'], {}


class BreastDataset(Dataset):
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

colours = ['\033[32m', '\033[33m', '\033[34m', '\033[35m','\033[36m', '\033[37m', '\033[90m', '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
reset = '\033[0m'
bold = '\033[01m'
disable = '\033[02m'
underline = '\033[04m'
reverse = '\033[07m'
strikethrough = '\033[09m'
invisible = '\033[08m'
default='\033[00m'
# print(
#     f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}"
# )

def cprint(text,id):
    print(f'{colours[id%13]} {text}{default}')
    

