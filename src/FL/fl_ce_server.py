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

from collections import OrderedDict
from typing import Dict, List, Tuple
from itertools import chain, combinations, permutations
from pathlib import Path
import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32
ROUND_TRUNC_THRESHOLD = 0.0
EPSILON = 0.0
CONVERGE_MIN_K = 30
CONVERGE_CRITERIA = 0.05
        
class CEServer(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, trainset, testset,id) -> None:
        super().__init__()
        self.model = model
        self.trainset = trainset
        self.testset = testset
        self.id = id
        self.gradients = []
        self.Contribution_records=[]
        self.last_k=10
         
    def identify(self):
        return True
        
    def utility(self, S):
        if S == ():
            accuracy, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
            return float(accuracy)
        l = len(S)
        params_model = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        gradient_sum = self.gradients[S[0]]
        for i in range(1,l):
            gradient_sum = [gradient_sum[j] + self.gradients[S[i]][j] for j in range(len(gradient_sum))]
        gradient_sum = [x/l for x in gradient_sum]
        parameters = [params_model[k] + gradient_sum[k] for k in range(len(gradient_sum))]
        temp_model = copy.deepcopy(self.model)
        params_dict = zip(temp_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        temp_model.load_state_dict(state_dict, strict=True)
        accuracy, loss, y_pred = GNN_script.test(temp_model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        return float(accuracy)
    
    def isnotconverge(self, k):
        if k <= CONVERGE_MIN_K:
            return True
        all_vals=(np.cumsum(self.Contribution_records, 0)/
                  np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-self.last_k:]
        errors = np.mean(np.abs(all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > CONVERGE_CRITERIA:
            return True
        return False
        
    def powersettool(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))
    
    def get_contributions2(self, gradients):
        self.gradients = gradients
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
        return SVs
        
    def get_contributions(self, gradients):
        self.gradients = gradients
        self.Contribution_records=[]
        N = len(gradients)   # number of participants
        idxs = [i for i in range(N)]
        util = {}  #record utilities for all group permutations
        
        S_0 = ()    # initial model = empty set
        util[S_0] = self.utility(S = S_0)     # v0 = V(Mt)  initial model utility
        S_all = list(self.powersettool(idxs))[-1]  # final model = full set of participants
        util[S_all] = self.utility(S = S_all)       # vN = V(Mt+1)  final model utility 
        
        myshapley_value = [0 for i in range(N)]
        
        if abs(util[S_all]-util[S_0]) <= ROUND_TRUNC_THRESHOLD:  # between round truncation
            return myshapley_value

        k=0
        while self.isnotconverge(k):
            for pi in idxs:
                k+=1
                v=[0 for i in range(N+1)]  # vk[0,...,N]  utilities of future models
                v[0]=util[S_0]             # vk,0 = v0
                marginal_contribution_k=[0 for i in range(N)] # marginal contributions at iteration k
                idxs_k=np.concatenate((np.array([pi]),np.random.permutation([p for p in idxs if p !=pi])))  #pik partial permutation

                for j in range(1,N+1):
                    # key = C subset
                    C=idxs_k[:j]      # participants subset of size j
                    C=tuple(np.sort(C,kind='mergesort'))

                    #truncation
                    if abs(util[S_all] - v[j-1]) >= EPSILON:         # within round truncation (remaining marginal gain small = |vN-vk,j-1|)
                        if util.get(C)!=None:
                            v[j]=util[C]           # if vk,j already computed 
                        else:
                            v[j]=self.utility(S=C)    # vk,j = utility of model with gradiants updates from participant set C
                    else:                           # here truncation because gain too small
                        v[j]=v[j-1]

                    util[C] = v[j]         # record calculated V(C)

                    marginal_contribution_k[idxs_k[j-1]-1] = v[j] - v[j-1]
                    myshapley_value[idxs_k[j-1]] = ((k-1)/k)*myshapley_value[idxs_k[j-1]] + (1/k)*(v[j] - v[j-1])
                self.Contribution_records.append(marginal_contribution_k)
                
        sv2 = self.get_contributions2(gradients)

        print("myshapley_value",myshapley_value)
        print("sv2",sv2)             

        return myshapley_value
        
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
        parameters = self.reshape_parameters(parameters)
        self.set_parameters(parameters)
        accuracy, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}
        
    
    
    
def main() -> None:

    # Parse command line argument `partition` and `nclients`
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
    args = parser.parse_args()
    n_clients = args.nclients
    id = n_clients


    #Dataset Loading
    families = ["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
    mapping = read_mapping("./mapping.txt")
    reversed_mapping = read_mapping_inverse("./mapping.txt")
    dataset, label, fam_idx, fam_dict, dataset_wl = GNN_script.init_dataset("./databases/examples_samy/BODMAS/01", families, reversed_mapping, [], {}, False)
    train_idx, test_idx = GNN_script.split_dataset_indexes(dataset, label)
    full_train_dataset,y_full_train, test_dataset,y_test = GNN_script.load_partition(n_clients=n_clients,id=id,train_idx=train_idx,test_idx=test_idx,dataset=dataset,client=False)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)


    #Model
    batch_size = 32
    hidden = 64
    num_classes = len(families)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    client = CEServer(model, full_train_dataset, test_dataset,id)
    #torch.save(model, f"HE/GNN_model.pt")
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())

if __name__ == "__main__":
    main()
