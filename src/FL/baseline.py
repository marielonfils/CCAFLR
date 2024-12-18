from flwr.common import NDArrays, Scalar
import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE

import flwr as fl
import numpy as np
import torch
import argparse
import copy

import SemaClassifier.classifier.GNN.GNN_script as GNN_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse

import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
import  SemaClassifier.classifier.GNN.gnn_helpers.models_training as models_training
import  SemaClassifier.classifier.GNN.gnn_helpers.models_tuning as models_tuning

from collections import OrderedDict
from typing import Dict, List, Tuple

from pathlib import Path
import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts

import SemaClassifier.classifier.GNN.gnn_main_script as main_script
import json
import time
import main_utils

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

class BaseClient():
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, trainset, testset,y_test,id,pk=None,filename=None ) -> None:
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.id=id
        self.context = None
        self._sk = None
        self.pk = None
        self.parms = None
        self.n = None
        self.shapes = None
        self.global_model = model
        
        self.y_test= y_test
        self.filename = filename
        self.train_time = 0

        self.set_context(8192,[60, 40, 40, 60] 	,2**40,pk)
    
    def set_context(self, poly_mod_degree, coeff_mod_bit_sizes,scale,pk=None):

        if pk is None: #generate random a for pk
            self.context = ts.context(ts.SCHEME_TYPE.MK_CKKS,poly_mod_degree,-1,coeff_mod_bit_sizes)
        else: #reuse a for pk
            self.context = ts.context(ts.SCHEME_TYPE.MK_CKKS,poly_mod_degree,-1,coeff_mod_bit_sizes,public_key=pk)
        self._sk = self.context.secret_key()
        self.context.generate_galois_keys(self._sk)
        self.context.make_context_public()
        self.context.global_scale = scale
        self.pk = self.context.public_key()
        self.shapes = [x.shape for x in self.get_parameters(config={})]
        
    def get_context(self):
        return self.context
    
    def get_pk(self):
        # return Ciphertext
        return self.pk
    
    def set_pk(self,pk):
        self.context.data.set_publickey(ts._ts_cpp.PublicKey(pk.data.ciphertext()[0]))
        self.pk = self.context.public_key()
    
    def encrypt(self, plain_vector):
        return ts.ckks_vector(self.context,np.array(plain_vector,dtype=object).flatten())
    
    def get_decryption_share(self, encrypted_vector):
        return self.context,encrypted_vector.decryption_share(self.context,self._sk)  

    def get_parameters(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def get_parameters_flat(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy().flatten() for _, val in self.model.state_dict().items()]
    
    def get_parms_enc(self, train=False) -> List[np.ndarray]:
        parms =  self.get_parameters_flat(config={})
        parms_flat = np.hstack(np.array(parms,dtype=object))
        if train:
            parms_flat = parms_flat*len(self.trainset)
        return self.context,self.encrypt(parms_flat)

    def get_gradients(self) -> List[np.ndarray]:
        print("##########   COMPUTING GRADIENT  #################")
        params_model1 = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        params_model2 = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        gradient = [params_model2[i] - params_model1[i] for i in range(len(params_model1))]
        return gradient
    
    def set_parameters(self, parameters: List[np.ndarray], N:int) -> None:
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def reshape_parameters(self,parameters):
        p=[]
        offset=0
        for _,s in enumerate(self.shapes):
            n = int(np.prod(s))
            if not s:
                p.append(np.array(parameters[offset],dtype=object))
            else:
                p.append(np.array(parameters[offset:(offset+n)],dtype=object).reshape(s))
            offset+=n
        return np.array(p,dtype=object)
    
    def fit(self, parameters: List[np.ndarray], config:Dict[str,str], flat=False) -> Tuple[List[np.ndarray], int, Dict]:
        #self.set_parameters(parameters, config["N"])
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        return self.get_parameters(config={}), len(self.trainset), loss    
    
    def fit_enc(self, parameters: List[np.ndarray], config:Dict[str,str],flat=True) -> Tuple[List[np.ndarray], int, Dict]:
        if flat:
            parameters = self.reshape_parameters(parameters)
        self.set_parameters(parameters, config)
        self.global_model = copy.deepcopy(self.model)
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        self.train_time=loss["train_time"]
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!SELF_TRAIN_TIME",self.train_time)
        main_utils.cprint(f"Client {self.id}: Fitting loss, {loss}", self.id)
        return self.get_parameters(config={}), len(self.trainset), loss

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        N=config.get("N")
        if N is None:
            N=1
        #self.set_parameters(parameters,N)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time], self.filename)
        main_utils.cprint(f"Client {self.id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss),"test_time": float(test_time),"train_time":float(self.train_time)}
    
    
    def evaluate_enc(self, parameters: List[np.ndarray], reshape = False
    ) -> Tuple[float, int, Dict]:
        if reshape:
            parameters = self.reshape_parameters(parameters)
            self.set_parameters(parameters,1)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time], self.filename)
        main_utils.cprint(f"Client {self.id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss),"test_time": float(test_time),"train_time":float(self.train_time)}
    
    
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
        type=str,
        required=False,
        help="Specifies the path for te dataset"
    )

    args = parser.parse_args()
    n_clients = args.nclients
    id = args.partition
    filename = args.filepath
    dataset_name = args.dataset
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename = f"{filename}/{timestr2}/client{id}_{timestr1}.csv"
    print("FFFNNN",filename)


    #Dataset Loading
    if dataset_name == "scdg1":
        ds_path = "./databases/scdg1"
        families=os.listdir(ds_path)
        mapping = read_mapping("./mapping_scdg1.txt")
        reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
    else:
        ds_path = "./databases/examples_samy/BODMAS/01"
        families=["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
        mapping = read_mapping("./mapping.txt")
        reversed_mapping = read_mapping_inverse("./mapping.txt")
        
    full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_all_datasets2(ds_path, families, mapping, reversed_mapping, n_clients, id)
    main_utils.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)
    

    #Model
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    #model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    client = BaseClient(model, full_train_dataset, test_dataset,y_test,id, filename=filename)
    for i in range(5):
        client.fit([],{})
        client.evaluate([],{})
    return filename
if __name__ == "__main__":
    main()
