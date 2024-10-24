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


#Dataset class
class Dataset:
    def __init__(self, init_db_function):
        self.init_db_function = init_db_function
        self.trainset = None #list or pytorch Dataset containing the training data
        self.y_train = None # list containing the training labels
        self.testset = None # list or pytorch Dataset containing the test data
        self.y_test = None # list containing the test labels
        self.classes = None # list containing the classes names
        self.others = None # dictionary with other return values
    
    def init_db(self, *args, **kwargs):
        # calls the init_db_function
        self.trainset, self.y_train, self.testset, self.y_test, self.classes, self.others=  self.init_db_function(*args, **kwargs)
    


def init_datasets(dataset_name, n_clients, id):
    # Initialize datasets
    # Arguments:
    #   dataset_name: str, the name of the dataset
    #   n_clients: int, the number of clients
    #   id: int, the id of the client
    # Returns a Dataset object
    
    # Modify here:
    # 1. create a branch for your datasetname
    # 2. create a Dataset with an db initialization function returning:
    #   - trainset : list or pytorch Dataset containing the training data, 
    #   - train labels : list containing the training labels, 
    #   - testset : list or pytorch Dataset containing the test data,
    #   - test labels : list containing the test labels,
    #   - classes : list containing the classes names,
    #   - others : dictionary with other return values
    # 3. call the init_db function with the correct arguments
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
    # Arguments:
    #   model_type: str, the type of model to use
    #   families: list, the list of classes names
    #   full_train_dataset: list or pytorch dataset, the training dataset
    #   model_path: str, the path to the model to load
    # Returns a Model object

    # Modify the parameters
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    model=None
    params={"batch_size":batch_size,"hidden":hidden,"num_classes":num_classes,"num_layers":num_layers,"drop_ratio":drop_ratio,"residual":residual,"model_type":model_type}
    
    #Modify here :
    #  - 1. create a branch for your model type
    #  - 2. create a model with the correct parameters
    #  - 3. create a model class with the correct train-test functions and parameters:
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
    #        The parameters are a dictionary containing the model parameters (batch_size, hidden, num_classes, num_layers, drop_ratio, residual, model_type)
    if model_path is not None: #load model
        model = torch.load(model_path,map_location=DEVICE)
        model.eval()
    else: #initialize model
        if model_type == "GINJKFlag":
            model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
            m = Model(model, gc.train, gc.test,params)
            return m
        elif model_type == "GINE":
            model = GINE(hidden, num_classes, num_layers).to(DEVICE)
            m = Model(model, gc.train, gc.test,params)
            return m
        elif model_type == "images":
            model=ConvNet(14)
            m = Model(model, ic.train, ic.test,params)
            return m
        elif model_type == "mobilenet":
            model=MobileNet(0.1,0.7,num_classes=2)
            m = Model(model, bc.train, bc.test,params)
            return m
    
    return model

#Model class
class Model:
    def __init__(self, model, train, test, params,get_model=None):
        self.model=model #pytorch model
        self.train=train #function for training the model
        self.test=test #function for evaluating the model
        self.get_model=get_model 
        self.parameters=params #dictionary containing the model parameters
    
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