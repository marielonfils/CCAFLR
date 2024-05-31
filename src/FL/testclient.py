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


DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
BATCH_SIZE=16
EPOCHS=5
BATCH_SIZE_TEST=32

def main() -> None:
    n_clients = 8
    id = 0
    dataset_name = "split_scdg1"
    model_path = "model_server_59.pt"
    
    ds_path = "./databases/scdg1"
    families=os.listdir(ds_path)
    mapping = read_mapping("./mapping_scdg1.txt")
    reversed_mapping = read_mapping_inverse("./mapping_scdg1.txt")
    full_train_dataset, y_full_train, test_dataset, y_test, label, fam_idx = main_script.init_split_dataset(mapping, reversed_mapping, n_clients, id)
    GNN_script.cprint(f"Client {id} : datasets length, {len(full_train_dataset)}, {len(test_dataset)}",id)
    
    model = torch.load(model_path)

    test_time, loss, y_pred = GNN_script.test(model, test_dataset, BATCH_SIZE_TEST, DEVICE,id)
    acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
    GNN_script.cprint(f"Client {id}: loss {loss}, accuracy {acc}, precision {prec}, recall {rec}, f1-score {f1}, balanced accuracy {bal_acc}", id)
        
    
if __name__ == "__main__":
    main()
    
