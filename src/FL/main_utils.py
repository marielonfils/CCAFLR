import argparse
import os
import sys
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
sys.path.insert(0, cwd+"/SemaClassifier/classifier/Images/")
import SemaClassifier.classifier.GNN.gnn_main_script as main_script
from SemaClassifier.classifier.Breast import breast_classifier as bc
from SemaClassifier.classifier.Images import ImageClassifier  as ic
import SemaClassifier.classifier.GNN.GNN_script as gc
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
from SemaClassifier.classifier.Breast.breast_classifier import MobileNet
from SemaClassifier.classifier.Images.ImageClassifier import split,ConvNet,ImagesDataset
import torch





DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



class Dataset:
    def __init__(self, init_db_function):
        self.init_db_function = init_db_function
        self.trainset = None
        self.y_train = None
        self.testset = None
        self.y_test = None
        self.classes = None
        self.others = None
    
    def init_db(self, *args, **kwargs):
        self.trainset, self.y_train, self.testset, self.y_test, self.classes, self.others=  self.init_db_function(*args, **kwargs)
    


def init_datasets(dataset_name, n_clients, id):
    # Initialize datasets
    # Modify here:
    # create a branch for your datasetname
    # create a Dataset with an db initialization function returning:
    #   - trainset : list or pytorch Dataset containing the training data, 
    #   - train labels : list containing the training labels, 
    #   - testset : list or pytorch Dataset containing the test data,
    #   - test labels : list containing the test labels,
    #   - classes : list containing the classes names,
    #   - others : dictionary with other return values
    # call the init_db function with the correct arguments
    if dataset_name=="scdg1": #scdg
        d=Dataset(gc.init_datasets_scdg1)
        d.init_db(n_clients,id)
        return d
    
    elif dataset_name == "split_scdg1": #scdg
        d=Dataset(gc.init_datasets_split_scdg1)
        d.init_db(n_clients, id)
        return d
    elif dataset_name == "images": #malware images
        d=Dataset(ic.init_datasets_images)
        d.init_db(n_clients, id)
        return d
    elif dataset_name =="breast": #breast images
        d = Dataset(bc.init_datasets_breast)
        d.init_db(n_clients,id)
        return d
    else: #scdg
        d=Dataset(gc.init_datasets_else)
        d.init_db(n_clients, id)
        return d
    


def get_model(model_type,families,full_train_dataset,model_path=None):
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    model=None
    
    #Modify here :
    #  - create a branch for your model type
    #  - create a model with the correct parameters
    #  - create a model with the correct train and test functions:
    #        The train function takes as arguments:
    #           - the model
    #           - the training dataset
    #          - the batch size
    #          - the number of epochs
    #          - the type of device
    #          - the id of the client
    #        The train function should return:
    #          - the trained model
    #          - a dictionary containing the training metrics ("loss", "train_time", ...)
    #        The test function takes as arguments:
    #          - the model
    #          - the test dataset
    #          - the batch size
    #          - the type of device
    #          - the id of the client
    #        The test function should return:
    #          - the test time
    #          - the test loss
    #          - the predicted labels (list)
    if model_path is not None: #load model
        model = torch.load(model_path,map_location=DEVICE)
        model.eval()
    else: #initialize model
        if model_type == "GINJKFlag":
            model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
            m = Model(model, gc.train, gc.test)
            return m
        elif model_type == "GINE":
            model = GINE(hidden, num_classes, num_layers).to(DEVICE)
            m = Model(model, gc.train, gc.test)
            return m
        elif model_type == "images":
            model=ConvNet(14)
            m = Model(model, ic.train, ic.test)
            return m
        elif model_type == "mobilenet":
            model=MobileNet(0.1,0.7,num_classes=2)
            m = Model(model, bc.train, bc.test)
            return m
    
    return model


class Model:
    def __init__(self, model, train, test, get_model=None):
        self.model=model
        self.train=train
        self.test=test
        self.get_model=get_model
    
    def get_model(self):
        if get_model is not None:
            return self.get_model()
        return self.model
    
    def set_model(self,model):
        self.model=model
    
    def train(self, *args, **kwargs):
        return self.train(*args, **kwargs)
    
    def test(self, *args, **kwargs):
        return self.test(*args, **kwargs)

colours = ['\033[32m', '\033[33m', '\033[34m', '\033[35m','\033[36m', '\033[37m', '\033[90m', '\033[91m', '\033[92m', '\033[93m', '\033[94m', '\033[95m', '\033[96m']
reset = '\033[0m'
bold = '\033[01m'
disable = '\033[02m'
underline = '\033[04m'
reverse = '\033[07m'
strikethrough = '\033[09m'
invisible = '\033[08m'
default='\033[00m'
def cprint(text,id):
    print(f'{colours[id%13]} {text}{default}')