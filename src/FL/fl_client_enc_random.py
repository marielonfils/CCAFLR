from flwr.common import NDArrays, Scalar
import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
from AESCipher import AESCipher
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
import random

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32
AESKEY = "bzefuilgfeilb4545h4rt5h4h4t5eh44eth878t6e738h"
class GNNClient(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, trainset, testset,y_test,id,pk=None,filename=None ) -> None:
        super().__init__()
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
        self.round = 0
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
        self.round += 1
        parms_flat = np.hstack(np.array(parms,dtype=object))
        if train:
            parms_flat = parms_flat#*len(self.trainset)
        return self.context,self.encrypt(parms_flat)

    def get_gradients(self):
        print("##########   COMPUTING GRADIENT  #################")
        #params_model1 = [val.cpu().numpy() for _, val in self.global_model.state_dict().items()]
        params_model2 = [np.array([self.id])] + [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        #gradient = [params_model2[i] - params_model1[i] for i in range(len(params_model1))]
        return AESCipher(AESKEY).encrypt(params_model2)
    
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
    
    def update_random_parameters(self):
        print("RANDOM CLIENT " + str(self.id))
        parameters = self.get_parameters(config={})
        for i in range(len(parameters)):
            sub_array = parameters[i]
            if isinstance(parameters[i], np.ndarray) and parameters[i].ndim > 0:
                for j in range(len(parameters[i])):
                    if isinstance(parameters[i][j], np.ndarray) and parameters[i][j].ndim > 0:
                        for k in range(len(parameters[i][j])):
                            random_float = random.gauss(0.0,0.4)
                            parameters[i][j][k] = max(np.float32(-1.0), min(np.float32(1.0), parameters[i][j][k]+random_float))
                    else:
                        random_float = random.gauss(0.0,0.4)
                        parameters[i][j] = max(np.float32(-1.0), min(np.float32(1.0), parameters[i][j]+random_float))
            else:
                random_float = random.gauss(0.0,0.4)
                parameters[i] = max(np.float32(-1.0), min(np.float32(1.0), parameters[i]+random_float))
        self.set_parameters(parameters,1)
        return
        
    def fit(self, parameters: List[np.ndarray], config:Dict[str,str], flat=False) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters, config["N"])
        if self.round > 10:
            self.update_random_parameters()
            return self.get_parameters(config={}), len(self.trainset), {}
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        return self.get_parameters(config={}), len(self.trainset), loss    
        
    
    def fit_enc(self, parameters: List[np.ndarray], config:Dict[str,str],flat=True) -> Tuple[List[np.ndarray], int, Dict]:
        if not(parameters != None and len(parameters) == 1):
            if flat:
                parameters = self.reshape_parameters(parameters)
            self.set_parameters(parameters, config)
        if self.round > 10:
            self.update_random_parameters()
            return self.get_parameters(config={}), len(self.trainset), {}    
        self.global_model = copy.deepcopy(self.model)
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        self.train_time=loss["train_time"]
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!SELF_TRAIN_TIME",self.train_time)
        GNN_script.cprint(f"Client {self.id}: Fitting loss, {loss}", self.id)
        return self.get_parameters(config={}), len(self.trainset), loss

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        N=config.get("N")
        if N is None:
            N=1
        self.set_parameters(parameters,N)
        accuracy, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        GNN_script.cprint(f"Client {self.id}: Evaluation accuracy & loss, {accuracy}, {loss}", self.id)
        #GNN_script.cprint(f"{self.id},{accuracy},{loss}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
    
    def evaluate_enc(self, parameters: List[np.ndarray], reshape = False
    ) -> Tuple[float, int, Dict]:
        if parameters != None and len(parameters) == 1:
            return 0.0,len(self.testset),{}
        if reshape:
            parameters = self.reshape_parameters(parameters)
            self.set_parameters(parameters,1)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time], self.filename)
        GNN_script.cprint(f"Client {self.id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", self.id)
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
        default = "",
        type=str,
        required=False,
        help="Specifies the path for te dataset"
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
    filename = args.filepath
    dataset_name = args.dataset
    model_path = args.modelpath
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename = f"{filename}/{timestr2}/random_client{id}_{timestr1}.csv"
    print("FFFNNN",filename)


    #Dataset Loading
    if "scdg1" in dataset_name:
        ds_path = "./databases/scdg1"
        families=os.listdir(ds_path)
        mapping = read_mapping("./mapping_scdg1.txt")
        reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
    else:
        ds_path = "./databases/examples_samy/BODMAS/01"
        families=["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
        mapping = read_mapping("./mapping.txt")
        reversed_mapping = read_mapping_inverse("./mapping.txt")
        
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
    client = GNNClient(model, full_train_dataset, test_dataset,y_test,id, filename=filename)
    #torch.save(model, f"HE/GNN_model.pt")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())
    return filename
if __name__ == "__main__":
    main()
