from flwr.common import NDArrays, Scalar
import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
from SemaClassifier.classifier.GNN.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.GINEClassifier import GINE

import flwr as fl
import numpy as np
import torch
import argparse

import SemaClassifier.classifier.GNN.GNN_script as GNN_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse

from collections import OrderedDict
from typing import Dict, List, Tuple

from pathlib import Path
import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

class GNNClient(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, trainset, testset,id,pk=None ) -> None:
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
        self.shape = None
        self.shapes = None
        self.length=None

        self.set_context(8192,[60,40,40,60],2**40,pk)
    
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
        self.shape = np.array(self.get_parameters(config={}),dtype=object).shape
        self.parms = np.hstack(np.array(self.get_parameters_flat(),dtype=object),dtype=object)
        self.length = len(self.parms)
        self.shapes = [x.shape for x in self.get_parameters(config={})]
        
    def get_context(self):
        return self.context
    
    def get_pk(self):
        # return Ciphertext
        return self.pk
    
    def set_pk(self,pk):
        self.context.data.set_publickey(ts._ts_cpp.PublicKey(pk.data.ciphertext()[0]))
        self.pk = self.context.public_key()
    
    def set_parms(self,parms,n):
        self.parms = parms 
        self.n = n 
    
    def get_parms(self):
        return self.parms

    def get_n(self):
        return self.n 
    
    def encrypt(self, plain_vector):
        return ts.ckks_vector(self.context,np.array(plain_vector,dtype=object).flatten())
    
    def get_decryption_share(self, encrypted_vector):
        return self.context,encrypted_vector.decryption_share(self.context,self._sk)  
    
    def decrypt(self, decryption_share):
        if self.ds is None or self.parms is None:
            raise ValueError("No decryption share available")
        return self.parms.mk_decrypt([decryption_share]) #TODO check if list is necessary

    def get_parameters(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
    
    def get_parameters_flat(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy().flatten() for _, val in self.model.state_dict().items()]
    
    def get_parms_enc(self) -> List[np.ndarray]:
        parms =  self.get_parameters_flat(config={})
        parms_flat = np.hstack(np.array(parms,dtype=object))
        return self.context,self.encrypt(parms_flat)
    
    def get_lenght(self):
        return self.length
    
    def set_parameters(self, parameters: List[np.ndarray], N:int) -> None:
        self.model.train()
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def reshape_parameters(self,parameters):
        p=[]
        offset=0
        for i,s in enumerate(self.shapes):
            GNN_script.cprint(f"i : {i}, s: {s}", 7)
            n = np.prod(s)
            p.append(np.array(parameters[offset:(offset+n)],dtype=object).reshape(s))
            offset+=n
        return np.array(p,dtype=object)

    def fit(self, parameters: List[np.ndarray], config:Dict[str,str], flat=False) -> Tuple[List[np.ndarray], int, Dict]:
        if flat:
            parameters = np.array(parameters,dtype=object).reshape(self.shape)
        self.set_parameters(parameters, config["N"])
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        return self.get_parameters(config={}), len(self.trainset), loss
    
    
    
    def fit_enc(self, parameters: List[np.ndarray], config:Dict[str,str],flat=True) -> Tuple[List[np.ndarray], int, Dict]:
        if flat:
            parameters = self.reshape_parameters(parameters)

        self.set_parameters(parameters, config)
        m, loss = GNN_script.train(self.model, self.trainset, BATCH_SIZE, EPOCHS, DEVICE, self.id)
        p = self.get_parameters(config={})
        return self.get_parameters(config={}), len(self.trainset), loss

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        N=config.get("N")
        if N is None:
            N=1
        self.set_parameters(parameters,N)
        accuracy, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        GNN_script.cprint(f"Client {self.id}: Evaluation accuracy & loss, {accuracy}, {loss}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
    
    def evaluate_enc(self, parameters: List[np.ndarray]
    ) -> Tuple[float, int, Dict]:
        parameters = self.reshape_parameters(self.parms)
        self.set_parameters(parameters,1)
        accuracy, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        GNN_script.cprint(f"Client {self.id}: Evaluation accuracy & loss, {accuracy}, {loss}", self.id)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
    
    
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
    args = parser.parse_args()
    n_clients = args.nclients
    id = args.partition


    #Dataset Loading
    families = ["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
    mapping = read_mapping("./mapping.txt")
    reversed_mapping = read_mapping_inverse("./mapping.txt")
    dataset, label, fam_idx, fam_dict, dataset_wl = GNN_script.init_dataset("./databases/examples_samy/BODMAS/01", families, reversed_mapping, [], {}, False)
    train_idx, test_idx = GNN_script.split_dataset_indexes(dataset, label)
    full_train_dataset,y_full_train, test_dataset,y_test = GNN_script.load_partition(n_clients=n_clients,id=id,train_idx=train_idx,test_idx=test_idx,dataset=dataset)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)


    #Model
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    # model = GINJKFlag(full_train_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    client = GNNClient(model, full_train_dataset, test_dataset,id)
    #torch.save(model, f"HE/GNN_model.pt")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())

if __name__ == "__main__":
    main()