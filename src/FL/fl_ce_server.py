import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
import SemaClassifier.classifier.GNN.GNN_script as GNN_script
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
import torch
import main_utils

import flwr as fl
from AESCipher import AESCipher
import rsa

import numpy as np
import copy
from collections import OrderedDict
from typing import Dict, List, Tuple
from itertools import chain, combinations, permutations
from pathlib import Path
import time


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
BATCH_SIZE_TEST=32

class CEServer(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, testset, y_test, id, enc, filename=None) -> None:
        super().__init__()
        self.t = time.time()
        self.model = model
        self.testset = testset
        self.y_test= y_test
        self.id = id
        self.gradients = []
        self.Contribution_records=[]
        self.last_k=10
        self.enc = enc
        self.filename=filename
        (self.publickey, self.privatekey) = rsa.newkeys(2048)
    
    def identify(self):
        return [str(self.publickey.n), str(self.publickey.e)]
        
    def utility(self, S):
        if S == ():
            test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
            accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
            return float(accuracy)
        l = len(S)
        gradient_sum = self.gradients[S[0]]
        for i in range(1,l):
            gradient_sum = [gradient_sum[j] + self.gradients[S[i]][j] for j in range(len(gradient_sum))]
        parameters = [x/l for x in gradient_sum]
        temp_model = copy.deepcopy(self.model)
        params_dict = zip(temp_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        temp_model.load_state_dict(state_dict, strict=True)
        test_time, loss, y_pred = GNN_script.test(temp_model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        return float(bal_acc)


    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        return
    
    def reshape_parameters(self,parameters):
        shapes = [x.shape for x in [val.cpu().numpy() for _, val in self.model.state_dict().items()]]
        p=[]
        offset=0
        for _,s in enumerate(shapes):
            n = int(np.prod(s))
            if not s:
                p.append(np.array(parameters[offset],dtype=object))
            else:
                p.append(np.array(parameters[offset:(offset+n)],dtype=object).reshape(s))
            offset+=n
        return np.array(p,dtype=object)
        
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]
    ) -> Tuple[float, int, Dict]:
        if self.enc:
            parameters = self.reshape_parameters(parameters)
        self.set_parameters(parameters)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
        
    def get_contributions(self, gradients):
        t1 = time.time()
        self.gradients = []
        mapping = {}
        for i,gradient in enumerate(gradients):
            AESKEY = rsa.decrypt(gradient[:256], self.privatekey).decode('utf8')
            parameters = AESCipher(AESKEY).decrypt(gradient[256:])
            self.gradients.append(self.reshape_parameters(parameters[1:]))
            mapping[parameters[0]] = i
        N = len(gradients)
        idxs = [i for i in range(N)]
        sets = list(chain.from_iterable(combinations(idxs, r) for r in range(len(idxs)+1)))
        util = {S:self.utility(S=S) for S in sets}
        SVs = [0 for i in range(N)]
        perms = list(permutations(idxs))
        for idx in idxs:
            SV = 0
            for t in perms:
                index = t.index(idx)
                u1 = util[tuple(sorted(t[:index]))]
                u2 = util[tuple(sorted(t[:index+1]))]
                SV += u2-u1
            SVs[idx] = SV/len(perms)
        t2 = time.time()
        c=[t2-t1]
        SV_sorted = [0 for i in range(self.id)]
        for m in mapping:
            SV_sorted[m] = SVs[mapping[m]]
        c.extend(SV_sorted)
        metrics_utils.write_contribution(c, self.filename)
        return SVs
        
def main() -> None:

    # Parse command line argument
    n_clients, id, filename, dataset_name, model_path, model_type,enc = main_utils.parse_arg_client()
    
    wo = ""
    if not enc:
        wo = "_wo"
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename = f"{filename}/{timestr2}{wo}/ce{id}_{timestr1}.csv"
    print("FFFNNN",filename)

    #Dataset Loading
    full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx, families, ds_path, mapping, reversed_mapping  =main_utils.init_datasets(dataset_name, n_clients, id)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)


    #Model
    model = main_utils.get_model(model_type, families,full_train_dataset,model_path)

    
    #Starting client
    client = CEServer(model, test_dataset, y_test, id, enc, filename)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())

if __name__ == "__main__":
    main()
