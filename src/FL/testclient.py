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
import random

#random.seed(42)
DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32


def update_random_parameters(model):
        print("RANDOM CLIENT ")
        parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
        for i in range(len(parameters)):
            sub_array = parameters[i]
            if isinstance(parameters[i], np.ndarray) and parameters[i].ndim > 0:
                for j in range(len(parameters[i])):
                    if isinstance(parameters[i][j], np.ndarray) and parameters[i][j].ndim > 0:
                        for k in range(len(parameters[i][j])):
                            random_float = random.gauss(0.0,0.02)
                            parameters[i][j][k] = parameters[i][j][k]+random_float
                    else:
                        random_float = random.gauss(0.0,0.02)
                        parameters[i][j] = parameters[i][j]+random_float
            else:
                random_float = random.gauss(0.0,0.02)
                parameters[i] = parameters[i]+random_float
        model.train()
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)
        return model
        
def main() -> None:
    n_clients = 8
    id = 0
    dataset_name = "split_scdg1"
    model_path = "models/model_server_30.pt"
    
    ds_path = "./databases/scdg1"
    families=os.listdir(ds_path)
    mapping = read_mapping("./mapping_scdg1.txt")
    reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
    full_train_dataset, y_full_train, x, y, z, w = main_script.init_split_dataset(mapping, reversed_mapping, n_clients, id)
    a, b, test_dataset, y_test, c, d = main_script.init_split_dataset(mapping, reversed_mapping, n_clients, 8)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)
    
    model = torch.load(model_path)

    test_time, loss, y_pred = GNN_script.test(model, test_dataset, BATCH_SIZE_TEST, DEVICE,8)
    acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
    GNN_script.cprint(f"Client {id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", id)
    for i in range(10):
        model = update_random_parameters(model)
        test_time, loss, y_pred = GNN_script.test(model, test_dataset, BATCH_SIZE_TEST, DEVICE,8)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
        GNN_script.cprint(f"Client {id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", id)
    
if __name__ == "__main__":
    main()
    
