import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
sys.path.insert(0, cwd+"/SemaClassifier/classifier/Images/")
sys.path.insert(0, cwd+"/SemaClassifier/classifier/Breast")
from SemaClassifier.classifier.Images import ImageClassifier  as img
from SemaClassifier.classifier.Breast import breast_classifier as bc
import SemaClassifier.classifier.GNN.GNN_script as GNN_script
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
from SemaClassifier.classifier.Breast.breast_classifier import MobileNet
import torch
import main_utils
import parse_args

from AESCipher import AESCipher
import secrets
import string
import rsa

import numpy as np
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple
from pathlib import Path
import time


sys.path.append('../../../../../TenSEAL')
import tenseal as ts

import flwr as fl



DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

class EncClient(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self,model_class, dataset,id,pk=None,filename=None, dirname=None ) -> None:
        super().__init__()
        #self.model = model
        self.d = dataset
        self.id=id
        self.shapes = None
        self.global_model = model_class.model
        self.round = 0
        self.m=model_class      

        #xMK CKKS context, sk, pk
        self.context = None
        self._sk = None
        self.pk = None
        self.set_context(8192,[60, 40, 40, 60] 	,2**40,pk)
                
        # File to save results
        self.filename = filename
        self.dirname = dirname
        self.train_time = 0

        #RSA public key
        self.publickey = ""
        
    
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
        return self.pk
    
    def set_pk(self,pk):
        self.context.data.set_publickey(ts._ts_cpp.PublicKey(pk.data.ciphertext()[0]))
        self.pk = self.context.public_key()
    
    def encrypt(self, plain_vector):
        return ts.ckks_vector(self.context,np.array(plain_vector,dtype=object).flatten())
    
    def get_decryption_share(self, encrypted_vector):
        return self.context,encrypted_vector.decryption_share(self.context,self._sk)  

    def get_parameters(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy() for _, val in self.m.model.state_dict().items()]
    
    def get_parameters_flat(self, config: Dict[str, str]=None) -> List[np.ndarray]:
        return [val.cpu().numpy().flatten() for _, val in self.m.model.state_dict().items()]
    
    def get_parms_enc(self, train=False) -> List[np.ndarray]:
        parms =  self.get_parameters_flat(config={})
        parms_flat = np.hstack(np.array(parms,dtype=object))
        if train:
            parms_flat = parms_flat#*len(self.trainset)
        return self.context,self.encrypt(parms_flat)

    def set_public_key(self, rsa_public_key):
        self.publickey = rsa.PublicKey(int(rsa_public_key[0]),int(rsa_public_key[1]))
        return
        
    def get_gradients(self):
        parameters = [np.array([self.id])] + [val.cpu().numpy() for _, val in self.m.model.state_dict().items()]
        characters = string.ascii_letters + string.digits + string.punctuation
        AESKEY = ''.join(secrets.choice(characters) for _ in range(245))
        encrypted_parameters = AESCipher(AESKEY).encrypt(parameters)
        encrypted_key = rsa.encrypt(AESKEY.encode('utf8'), self.publickey)
        return encrypted_key + encrypted_parameters
    
    def set_parameters(self, parameters: List[np.ndarray], N:int) -> None:
        self.m.model.train()
        params_dict = zip(self.m.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        self.m.model.load_state_dict(state_dict, strict=True)

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
        
    def fit_enc(self, parameters: List[np.ndarray], config:Dict[str,str],flat=True) -> Tuple[List[np.ndarray], int, Dict]:
        # set parameters
        if not(parameters != None and len(parameters) == 1):
            if flat:
                parameters = self.reshape_parameters(parameters)
            self.set_parameters(parameters, config)
        # save global model
        self.global_model = copy.deepcopy(self.m.model)
        if self.dirname is not None:
            torch.save(self.global_model,f"{self.dirname}/model_global_{self.round}.pt")
        self.round+=1
        #train model
        _,loss=self.m.train(self.m.model,self.d.trainset,EPOCHS,BATCH_SIZE,self.id,DEVICE)
        #save local model
        if self.dirname is not None:
            torch.save(self.m.model,f"{self.dirname}/model_local_{self.round}.pt")
        self.train_time=loss["train_time"]
        main_utils.cprint(f"Client {self.id}: Fitting loss, {loss}", self.id)

        return self.get_parameters(config={}), len(self.d.trainset), loss
    
    def evaluate_enc(self, parameters: List[np.ndarray], config=None, reshape = False
    ) -> Tuple[float, int, Dict]:
        
        if parameters != None and len(parameters.parameters.tensors) == 1: #TODO check condition parameters is a FitIns instance
            return 0.0,len(self.d.testset),{}
        # set parameters
        if reshape:
            parameters = self.reshape_parameters(parameters)
            self.set_parameters(parameters,1)
        #evaluate
        test_time, loss, y_pred = self.m.test(self.m.model,self.d.testset,BATCH_SIZE_TEST, self.id, DEVICE)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.d.y_test, y_pred)
        metrics_utils.write_to_csv([str(self.m.model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, self.train_time, test_time, str(np.array_str(np.array(y_pred),max_line_width=10**50))], self.filename)
        main_utils.cprint(f"Client {self.id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", self.id)
        return float(loss), len(self.d.testset), {"accuracy": float(acc),"precision": float(prec), "recall": float(rec), "f1": float(f1), "balanced_accuracy": float(bal_acc),"loss": float(loss),"test_time": float(test_time),"train_time":float(self.train_time)}


def main() -> None:

    # Parse command line arguments
    n_clients, id, filename, dataset_name, model_path, model_type, datapath, split = parse_args.parse_arg_client()

    dirname=None
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        dirname = f"{filename}/{timestr2}/parms_{id}/"
        filename = f"{filename}/{timestr2}/client{id}_{timestr1}.csv"
    print("FFFNNN",filename)

    if not os.path.isdir(dirname):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)


    #Dataset Loading
    #Modify the init_datasets function in main_utils
    d= main_utils.init_datasets(dataset_name, datapath, split, n_clients, id)
    main_utils.cprint(f"Client {id} : datasets length, {len(d.trainset)}, {len(d.testset)}",id)
    

    #Model
    #Modify model_type here and get_model function in main_utils
    m = main_utils.get_model(model_type, d.classes, d.trainset,model_path=model_path)

    
    #Client
    client = EncClient(m, d,id,  filename=filename, dirname=dirname)
    
    #torch.save(model, f"HE/GNN_model.pt")
    # Start Flower client
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())
    # Write results
    with open(filename,'a') as f:
        f.write(str(d.y_test)+"\n")
    return filename
    
if __name__ == "__main__":
    main()
