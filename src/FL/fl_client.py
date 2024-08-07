from flwr.common import NDArrays, Scalar

from pathlib import Path
import sys
import os
cwd=os.getcwd()
print(cwd,cwd+"/src")
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
sys.path.append('../../../../../TenSEAL')
import tenseal as ts
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
from AESCipher import AESCipher
import flwr as fl
import numpy as np
import torch
import argparse
import SemaClassifier.classifier.GNN.GNN_script as GNN_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse

from collections import OrderedDict
from typing import Dict, List, Tuple
import copy
import time
import SemaClassifier.classifier.GNN.gnn_main_script as main_script
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
import secrets
import string
import rsa

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

class GNNClient(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, trainset, testset,y_test,id,pk=None,filename=None, dirname=None ) -> None:
        super().__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.id=id
        self.y_test=y_test
        self.filename = filename
        self.dirname = dirname
        self.train_time = 0
        self.global_model = model
        self.publickey = ""
        self.round=0
    
    def set_public_key(self, rsa_public_key):
        self.publickey = rsa.PublicKey(int(rsa_public_key[0]),int(rsa_public_key[1]))
        return
  
    def get_parameters(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(self, parameters: List[np.ndarray], config:Dict[str,str]) -> Tuple[List[np.ndarray], int, Dict]:
        if "wait" in config and config["wait"] == "no_update":
                pass
        else:
            self.set_parameters(parameters)
        self.global_model = copy.deepcopy(self.model)
        torch.save(self.global_model,f"{self.dirname}/model_global_{self.round}.pt")
        self.round+=1
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        torch.save(self.model,f"{self.dirname}/model_local_{self.round}.pt")
        self.train_time=loss["train_time"]
        p = self.get_parameters(config={})
        l=loss["loss"]
        GNN_script.cprint(f"Client {self.id}: Evaluation accuracy & loss, {l}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", self.id)

        metrics_utils.write_to_csv([str(self.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss["loss"], self.train_time, test_time, str(np.array_str(np.array(y_pred),max_line_width=10**50))], self.filename)

        return self.get_parameters(config={}), len(self.trainset) ,{"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss["loss"]),"test_time": float(test_time),"train_time":float(self.train_time)}
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        #self.set_parameters(parameters)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time, str(np.array_str(np.array(y_pred),max_line_width=10**50))], self.filename)
        GNN_script.cprint(f"Client {self.id}: Evaluation accuracy & loss, {loss}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss),"test_time": float(test_time),"train_time":float(self.train_time)}

    def get_gradients(self):
        parameters = [np.array([self.id])] + [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        characters = string.ascii_letters + string.digits + string.punctuation
        AESKEY = ''.join(secrets.choice(characters) for _ in range(245))
        encrypted_parameters = AESCipher(AESKEY).encrypt(parameters)
        encrypted_key = rsa.encrypt(AESKEY.encode('utf8'), self.publickey)
        return encrypted_key + encrypted_parameters
    
    
def main() -> None:

    # Parse command line argument `partition` and `nclients`
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
    
    args = parser.parse_args()
    n_clients = args.nclients
    id = args.partition
    dataset_name = args.dataset
    filename = args.filepath
    dirname = ""
    model_path = args.modelpath
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        dirname = f"{filename}/{timestr2}_wo/parms_{id}/"
        filename = f"{filename}/{timestr2}_wo/client{id}_{timestr1}.csv"
    print("FFFNNN",filename)

    if not os.path.isdir(dirname):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)

    #Dataset Loading
    families=[0,1,2,3,4,5,6,7,8,9,10,11,12] #13 families in scdg1
    ds_path=""
    mapping = {}
    reversed_mapping = {}
    if "scdg1" in dataset_name:
        mapping = read_mapping("./mapping_scdg1.txt")
        reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
    else:
        ds_path = "./databases/examples_samy/BODMAS/01"
        families=["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
        mapping = read_mapping("./mapping.txt")
        reversed_mapping = read_mapping_inverse("./mapping.txt")
    
    if "scdg1" == dataset_name:
        ds_path = "./databases/scdg1"
        families=os.listdir(ds_path)
        
    if dataset_name == "split_scdg1":
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_split_dataset(mapping, reversed_mapping, n_clients, id)
    else:
        full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_all_datasets(ds_path, families, mapping, reversed_mapping, n_clients, id)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)
    

    #Model
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    #model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    
    if model_path is not None:
        model = torch.load(model_path)
    #Client
    client = GNNClient(model, full_train_dataset, test_dataset,y_test,id, filename=filename, dirname=dirname)
    #torch.save(model, f"HE/GNN_model.pt")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())
    with open(filename,'a') as f:
        f.write(str(y_test)+"\n")
    return filename
    
if __name__ == "__main__":
    main()
    
