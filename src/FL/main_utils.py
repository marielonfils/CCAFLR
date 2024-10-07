import argparse
import os
import sys
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
sys.path.insert(0, cwd+"/SemaClassifier/classifier/Images/")
import SemaClassifier.classifier.GNN.gnn_main_script as main_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
from SemaClassifier.classifier.Images.ImageClassifier import split,ConvNet,ImagesDataset
import torch
from torchvision.transforms import transforms

from SemaClassifier.classifier.Images import ImageClassifier  as img
import SemaClassifier.classifier.GNN.GNN_script as GNN_script


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_arg_server():
    # Parse command line argument of server
    parser = argparse.ArgumentParser(description="Flower")    
    parser.add_argument(
        "--nclients",
        type=int,
        default=1,
        choices=range(1, 10),
        required=False,
        help="Specifies the number of clients. \
        Picks partition 1 by default",
    )
    parser.add_argument(
        "--nrounds",
        type=int,
        default=3,
        choices=range(1, 100),
        required=False,
        help="Specifies the number of rounds of FL. \
        Picks partition 3 by default",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=False,
        help="Specifies the path for storing results"
    )
    parser.add_argument(
        "--dataset",
        default = "",
        type=str,
        required=False,
        help="Specifies the path for the dataset",
    )
    parser.add_argument(
        "--noce",
        action="store_false",
        help="Specifies if there is contribution evaluation or not",
    )

    parser.add_argument(
        "--methodo",
        default = "",
        type=str,
        required=False,
        help="Specifies the methodology used to deal with client that have low SV"
    )
    parser.add_argument(
        "--threshold",
        default = -1.0,
        type=float,
        required=False,
        help="Specifies the threshold to delete clients"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )

    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )
    
    args = parser.parse_args()
    n_clients = args.nclients
    id = n_clients
    nrounds = args.nrounds
    dataset_name = args.dataset
    methodo = args.methodo
    threshold = args.threshold
    filename = args.filepath
    ce=args.noce
    model_type = args.model
    model_path = args.modelpath

    return n_clients, id, nrounds, dataset_name, methodo, threshold, filename, ce, model_type, model_path

def parse_arg_client():
    # Parse command line argument of client
    parser = argparse.ArgumentParser(description="Flower")    
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the id of the client. \
        Picks partition 0 by default",
    )
    parser.add_argument(
        "--nclients",
        type=int,
        default=1,
        choices=range(1, 10),
        required=False,
        help="Specifies the number of clients for dataset partition. \
        Picks partition 1 by default",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=False,
        default="./results",
        help="Specifies the path for storing results"
    )
    parser.add_argument(
        "--dataset",
        default = "",
        type=str,
        required=False,
        help="Specifies the path for the dataset"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )

    args = parser.parse_args()
    n_clients = args.nclients
    id = args.partition
    filename = args.filepath
    dataset_name = args.dataset
    model_path = args.modelpath
    model_type = args.model

    return n_clients, id, filename, dataset_name, model_path, model_type

def parse_arg_ce():
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--nclients",
        type=int,
        default=1,
        choices=range(1, 10),
        required=False,
        help="Specifies the number of clients for dataset partition. \
        Picks partition 1 by default",
    )
    parser.add_argument(
        "--dataset",
        default = "",
        type=str,
        required=False,
        help="Specifies the path for the dataset",
    )
    parser.add_argument(
        "--filepath",
        type=str,
        required=False,
        help="Specifies the path for storing results"
    )
    parser.add_argument(
        "--enc",
        action="store_true",
        help="Specifies if there is encryption or not",
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )
    args = parser.parse_args()
    n_clients = args.nclients
    dataset_name = args.dataset
    id = n_clients
    enc = args.enc
    filename = args.filepath
    model= args.model
    model_path = args.modelpath

    return n_clients, id, filename,dataset_name,model,model_path,enc



def init_datasets(dataset_name, n_clients, id):
    # Initialize datasets
    families=[0,1,2,3,4,5,6,7,8,9,10,11,12] #13 families in scdg1
    ds_path=""
    mapping = {}
    reversed_mapping = {}

    if dataset_name=="scdg1": #scdg
        ds_path = "./databases/scdg1"
        families=os.listdir(ds_path)
        mapping = read_mapping("./mapping_scdg1.txt")
        reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_all_datasets(ds_path, families, mapping, reversed_mapping, n_clients, id)
    
    elif dataset_name == "split_scdg1": #scdg
        mapping = read_mapping("./mapping_scdg1.txt")
        reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_split_dataset(mapping, reversed_mapping, n_clients, id)
    
    elif dataset_name == "images": #images
        if n_clients == id:
            path = "./databases/Images/server"
        else:
            path = "./databases/Images/client"+str(id+1)
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = split(path)
        families=[0,1,2,3,4,5,6,7,8,9,10,11,12,13]
        transforms_train = transforms.Compose([transforms.Resize((128, 128)),
                                       transforms.RandomRotation(10.),
                                       transforms.ToTensor()])

        transforms_test = transforms.Compose([transforms.Resize((128, 128)),
                                            transforms.ToTensor()])
        full_train_dataset=ImagesDataset(full_train_dataset,y_full_train,transforms_train)
        test_dataset=ImagesDataset(test_dataset,y_test,transform=transforms_test)
    
    else: #scdg
        ds_path = "./databases/examples_samy/BODMAS/01"
        families=["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
        mapping = read_mapping("./mapping.txt")
        reversed_mapping = read_mapping_inverse("./mapping.txt")
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_all_datasets(ds_path, families, mapping, reversed_mapping, n_clients, id)

    return full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx, families, ds_path, mapping, reversed_mapping
 


def get_model(model_type,families,full_train_dataset,model_path=None):
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    model=None
    
    if model_path is not None: #load model
        model = torch.load(model_path,map_location=DEVICE)
        model.eval()
    else: #initialize model
        if model_type == "GINJKFlag":
            model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
        elif model_type == "GINE":
            model = GINE(hidden, num_classes, num_layers).to(DEVICE)
        elif model_type == "images":
            model=ConvNet(14)
    
    return model

def train(model_type, model, trainset, batch_size, epochs,id, device=None ):
    if model_type == "GINE":
        m, loss = GNN_script.train(model, trainset, batch_size,epochs,device,id)
    elif model_type == "images":
        m,loss=img.train(model,trainset,batch_size,epochs,id)
    return m,loss

def test(model_type, model, testset, batch_size, id, device=None):
    if model_type == "GINE":
        test_time, loss, y_pred = GNN_script.test(model, testset, batch_size, DEVICE,id)
    elif model_type == "images":
        test_time, loss, y_pred = img.test(model,testset,batch_size,id)
    return test_time, loss, y_pred

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