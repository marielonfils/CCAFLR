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
from AESCipher import AESCipher
import SemaClassifier.classifier.GNN.GNN_script as GNN_script
import SemaClassifier.classifier.GNN.gnn_main_script as main_script
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse

from collections import OrderedDict
from typing import Dict, List, Tuple
from itertools import chain, combinations, permutations
from pathlib import Path
import time
import sys
sys.path.append('../../../../../TenSEAL')
import tenseal as ts

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
BATCH_SIZE_TEST=32
AESKEY = "bzefuilgfeilb4545h4rt5h4h4t5eh44eth878t6e738h"

class CEServer(fl.client.NumPyClient):
    """Flower client implementing Graph Neural Networks using PyTorch."""

    def __init__(self, model, testset, y_test, id, enc, filename, filename2) -> None:
        super().__init__()
        self.t = time.time()
        self.model = model
        self.testset = testset
        self.y_test= y_test
        self.id = id
        self.gradients = []
        self.Contribution_records=[]
        self.last_k=10
        self.round = 0
        self.enc = enc
        self.filename=filename
        self.filename2=filename2
         
    def identify(self):
        return True
        
    def utility(self, S):
        if S == ():
            test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
            accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
            return float(accuracy)
        l = len(S)
        #params_model = [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        gradient_sum = self.gradients[S[0]]
        for i in range(1,l):
            gradient_sum = [gradient_sum[j] + self.gradients[S[i]][j] for j in range(len(gradient_sum))]
        #gradient_sum = [x/l for x in gradient_sum]
        #parameters = [params_model[k] + gradient_sum[k] for k in range(len(gradient_sum))]
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
        if "check" in config and config["check"] == False:
            print("CHECK IS FALSE")
            return float(0.0), len(self.testset), {"accuracy": float(0.0)}
        print("CHECK IS TRUE")
        self.set_parameters(parameters)
        test_time, loss, y_pred = GNN_script.test(self.model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        return float(loss), len(self.testset), {"accuracy": float(accuracy)}

    def accuracy_client(self, gradients):
        decrypt = [AESCipher(AESKEY).decrypt(gradient) for gradient in gradients]
        gradients = {d[0]:self.reshape_parameters(d[1:]) for d in decrypt}
        accuracies = {i:0 for i in range(8)}
        for c in gradients:
            parameters = gradients[c]
            temp_model = copy.deepcopy(self.model)
            params_dict = zip(temp_model.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
            temp_model.load_state_dict(state_dict, strict=True)
            test_time, loss, y_pred = GNN_script.test(temp_model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
            accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
            accuracies[c] = float(bal_acc)
            
        g_values = list(gradients.values())
        l = len(g_values)
        gradient_sum = g_values[0]
        for i in range(1,l):
            gradient_sum = [gradient_sum[j] + g_values[i][j] for j in range(len(gradient_sum))]
        parameters = [x/l for x in gradient_sum]
        temp_model = copy.deepcopy(self.model)
        params_dict = zip(temp_model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        temp_model.load_state_dict(state_dict, strict=True)
        test_time, loss, y_pred = GNN_script.test(temp_model, self.testset, BATCH_SIZE_TEST, DEVICE,self.id)
        accuracy, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(self.y_test, y_pred)
        
        with open(self.filename2,"a") as f:
            for i in range(len(accuracies)):
                f.write(str(accuracies[i]))
                f.write(",")
            f.write(str(float(bal_acc)))
            f.write("\n")
        return
        
    def get_contributions(self, gradients):
        if self.round == 0:
            with open(self.filename2,"a") as f:
                s = ""
                for i in range(8):
                    s += str(i)+","
                s += "all \n"
                f.write(s)
        self.accuracy_client(gradients)
        self.round += 1
        t1 = time.time()
        mapping = {AESCipher(AESKEY).decrypt(gradients[i])[0]:i for i in range(len(gradients))}
        self.gradients = [self.reshape_parameters(AESCipher(AESKEY).decrypt(gradient)[1:]) for gradient in gradients]
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
        c=["original_shapley",str(self.model.__class__.__name__),N, t2-t1]
        SV_sorted = [0 for i in range(N)]
        for m in mapping:
            SV_sorted[m] = SVs[mapping[m]]
        c.extend(SV_sorted)
        metrics_utils.write_contribution(c, self.filename)
        #self.get_contributions_gtg(gradients)
        print("Original shapley","time: "+str(t2-t1),SVs)
        return SVs
        
    def isnotconverge(self, k):
        if k <= 30:
            return True
        if k > 100:  # to avoid infinite loop
             return False
        all_vals=(np.cumsum(self.Contribution_records, 0)/
                  np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-self.last_k:]
        errors = np.mean(np.abs(all_vals[-self.last_k:] - all_vals[-1:])/(np.abs(all_vals[-1:]) + 1e-12), -1)
        if np.max(errors) > 0.05:
            return True
        return False
        
    def powersettool(self, iterable):
        "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
        s = list(iterable)
        return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

    def get_contributions_gtg(self, gradients):  #GTG shapley implem from https://github.com/liuzelei13/GTG-Shapley/tree/50879e8905aaf5f4d408961b69523cacc33deb47
        t1 = time.time()
        self.gradients = self.gradients = [self.reshape_parameters(AESCipher(AESKEY).decrypt(gradient)) for gradient in gradients]
        self.Contribution_records=[]
        N = len(gradients)   # number of participants
        idxs = [i for i in range(N)]
        util = {}  #record utilities for all group permutations
        
        S_0 = ()    # initial model = empty set
        util[S_0] = self.utility(S = S_0)     # v0 = V(Mt)  initial model utility
        S_all = list(self.powersettool(idxs))[-1]  # final model = full set of participants
        util[S_all] = self.utility(S = S_all)       # vN = V(Mt+1)  final model utility 
      
        if abs(util[S_all]-util[S_0]) <= 0.01:  # between round truncation
            t2 = time.time()
            SVs = [0 for i in range(N)]
            c=["gtg_shapley",str(self.model.__class__.__name__),N, t2-t1]
            c.extend(SVs)
            metrics_utils.write_contribution(c, self.filename)
            print("GTG shapley","time: "+str(t2-t1),SVs)
            return 

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
                    if abs(util[S_all] - v[j-1]) >= 0.001:         # within round truncation (remaining marginal gain small = |vN-vk,j-1|)
                        if util.get(C)!=None:
                            v[j]=util[C]           # if vk,j already computed 
                        else:
                            v[j]=self.utility(S=C)    # vk,j = utility of model with gradiants updates from participant set C
                    else:                           # here truncation because gain too small
                        v[j]=v[j-1]

                    util[C] = v[j]         # record calculated V(C)

                    marginal_contribution_k[idxs_k[j-1]-1] = v[j] - v[j-1]
                self.Contribution_records.append(marginal_contribution_k)
        shapley_value = (np.cumsum(self.Contribution_records, 0)/
                         np.reshape(np.arange(1, len(self.Contribution_records)+1), (-1,1)))[-1:].tolist()[0]
        SVs = [shapley_value[(i-1)%N] for i in range(N)]
        t2 = time.time()
        c=["gtg_shapley",str(self.model.__class__.__name__),N, t2-t1]
        c.extend(SVs)
        metrics_utils.write_contribution(c, self.filename)
        print("GTG shapley","time: "+str(t2-t1),SVs)
        return
    
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
    args = parser.parse_args()
    n_clients = args.nclients
    dataset_name = args.dataset
    id = n_clients
    enc = args.enc
    wo = ""
    if not enc:
        wo = "_wo"
    filename = args.filepath
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename2 = f"{filename}/{timestr2}{wo}/ce{id}_{timestr1}_accuracy_client.csv"
        filename = f"{filename}/{timestr2}{wo}/ce{id}_{timestr1}.csv"
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
    num_layers = 2
    drop_ratio = 0.5
    residual = False
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    
    #Starting client
    client = CEServer(model, test_dataset, y_test, id, enc, filename,filename2)
    fl.client.start_numpy_client(server_address="127.0.0.1:8080", client=client, root_certificates=Path("./FL/.cache/certificates/ca.crt").read_bytes())

if __name__ == "__main__":
    main()
