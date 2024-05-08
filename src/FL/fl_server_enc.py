"""Flower server example."""


import torch
import flwr as fl
from typing import Dict, Optional, Tuple
from collections import OrderedDict
import argparse

import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
from FL.CE_client_manager import CEClientManager
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")

from SemaClassifier.classifier.GNN import GNN_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse
from torch_geometric.loader import DataLoader
from SemaClassifier.classifier.GNN.models.GINJKFlagClassifier import GINJKFlag
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE

from pathlib import Path

import SemaClassifier.classifier.GNN.gnn_main_script as main_script
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
import time
import json

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

def reshape_parameters(parameters,shapes):
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


def fit_config(server_round: int):
    """Return training configuration dict for each round.

    Keep batch size fixed at 32, perform two rounds of training with one local epoch,
    increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config

def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds, one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}

def get_evaluate_fn(model: torch.nn.Module, valset,id):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # valLoader = DataLoader(valset, batch_size=16, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters

        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        accuracy, loss, y_pred  = GNN_script.test(model, valset, 32, DEVICE,id)
        #GNN_script.cprint(f"Server: Evaluation accuracy & loss, {accuracy}, {loss}",id)
        GNN_script.cprint(f"{id},{accuracy},{loss}",id)

        return loss, {"accuracy": accuracy}

    return evaluate

def get_evaluate_enc_fn(model: torch.nn.Module, valset,id,y_test):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # valLoader = DataLoader(valset, batch_size=16, shuffle=False)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        parameters = reshape_parameters(parameters,[x.cpu().numpy().shape for x in model.state_dict().values()])
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        test_time, loss, y_pred  = GNN_script.test(model, valset, 32, DEVICE,id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
        #metrics_utils.write_to_csv([str(model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, 0, 0,0,0], filename)
        GNN_script.cprint(f"Client {id}: Evaluation accuracy & loss, {loss}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", id)
        
        return loss, {"accuracy": acc,"precision": prec,"recall": rec,"f1": f1,"balanced_accuracy": bal_acc,"loss": loss, "test_time": test_time, "train_time":0}

    return evaluate

def get_aggregate_evaluate_enc_fn(model: torch.nn.Module, valset,id,metrics):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    # valLoader = DataLoader(valset, batch_size=16, shuffle=False)

    # The `evaluate` function will be called after every round
    def aggregate_evaluate(
        eval_metrics,
        #server_round: int,
        #parameters: fl.common.NDArrays,
        #config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters

        n_tot=0
        agg = {m:0 for m in metrics}
        #agg=[0 for _ in range(len(metrics))]
        for r in eval_metrics:
            n,m = r
            n_tot+=n
            for metric in metrics:
                agg[metric]+=m[metric]*n
        for metric in metrics:
            agg[metric]/=n_tot
        return agg
    return aggregate_evaluate

def main():
    #Parse command line argument `nclients`
    parser = argparse.ArgumentParser(description="Flower")    
    parser.add_argument(
        "--nclients",
        type=int,
        default=1,
        choices=range(1, 10),
        required=False,
        help="Specifies the number of clients. \
        Picks partition 1 by default",
    )
    parser.add_argument(
        "--nrounds",
        type=int,
        default=3,
        choices=range(1, 10),
        required=False,
        help="Specifies the number of rounds of FL. \
        Picks partition 3 by default",
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
    id = n_clients
    nrounds = args.nrounds
    filename = args.filepath
    dataset_name = args.dataset
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename1 = f"{filename}/{timestr2}/model.txt"
        filename = f"{filename}/{timestr2}/server{id}_{timestr1}.csv"
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
    # model = GINJKFlag(test_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
    model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]
    if filename is not None:
        metrics_utils.write_model(filename1,{"model":model.__class__.__name__,"batch_size":batch_size,"hidden":hidden,"num_classes":num_classes,"num_layers":num_layers,"drop_ratio":drop_ratio,"residual":residual,"device":DEVICE,"n_clients":n_clients,"id":id,"nrounds":nrounds,"filename":filename,"ds_path":ds_path,"families":families,"mapping":mapping,"reversed_mapping":reversed_mapping,"full_train_dataset":len(full_train_dataset),"test_dataset":len(test_dataset)})

    
    # FL strategy
    strategy = fl.server.strategy.MKFedAvg(
        fraction_fit=0.2,  # Fraction of available clients used for training at each round
        min_fit_clients=n_clients,#2,  # Minimum number of clients used for training at each round (override `fraction_fit`)
        min_evaluate_clients=n_clients,  # Minimum number of clients used for testing at each round 
        min_available_clients=n_clients,#2,  # Minimum number of all available clients to be considered
        evaluate_fn=get_evaluate_enc_fn(model, test_dataset, id,y_test),  # Evaluation function used by the server 
        evaluate_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(model, test_dataset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time"]),
        fit_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(model, test_dataset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time"]),
        on_fit_config_fn=fit_config,  # Called before every round
        on_evaluate_config_fn=evaluate_config,  # Called before evaluation rounds
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    client_manager = CEClientManager()
    
    # import pdb; pdb.set_trace()
    hist=fl.server.start_server(
        length=len(np.hstack(np.array([val.cpu().numpy().flatten() for _, val in model.state_dict().items()],dtype=object),dtype=object)),
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=nrounds),
        strategy=strategy,
        client_manager=client_manager,
        certificates=(
            Path("./FL/.cache/certificates/ca.crt").read_bytes(),
            Path("./FL/.cache/certificates/server.pem").read_bytes(),
            Path("./FL/.cache/certificates/server.key").read_bytes(),),
        enc=True,
    )
    if filename is not None:
        metrics_utils.write_history_to_csv(hist,model, nrounds, filename)
    return

if __name__ == "__main__":
    main()
