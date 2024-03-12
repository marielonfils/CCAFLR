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
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")

from SemaClassifier.classifier.GNN import GNN_script
from SemaClassifier.classifier.GNN.utils import read_mapping, read_mapping_inverse
from torch_geometric.loader import DataLoader
from SemaClassifier.classifier.GNN.GINJKFlagClassifier import GINJKFlag

from pathlib import Path

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
import numpy as np

def reshape_parameters(parameters,shapes):
    p=[]
    offset=0
    for i,s in enumerate(shapes):
        n = np.prod(s)
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
        GNN_script.cprint(f"Server: Evaluation accuracy & loss, {accuracy}, {loss}",id)

        return loss, {"accuracy": accuracy}

    return evaluate

def get_evaluate_enc_fn(model: torch.nn.Module, valset,id):
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

        accuracy, loss, y_pred  = GNN_script.test(model, valset, 32, DEVICE,id)
        GNN_script.cprint(f"Server: Evaluation accuracy & loss, {accuracy}, {loss}",id)

        return loss, {"accuracy": accuracy}

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


        parameters = reshape_parameters(parameters,[x.cpu().numpy().shape for x in model.state_dict().values()])
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        accuracy, loss, y_pred  = GNN_script.test(model, valset, 32, DEVICE,id)
        GNN_script.cprint(f"Server: Evaluation accuracy & loss, {accuracy}, {loss}",id)

        return loss, {"accuracy": accuracy}

    return aggregate_evaluate

if __name__ == "__main__":
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
    args = parser.parse_args()
    n_clients = args.nclients
    id = n_clients
    nrounds = args.nrounds
    
    #Dataset loading
    families = ["berbew","sillyp2p","benjamin","small","mira","upatre","wabot"]
    mapping = read_mapping("./mapping.txt")
    reversed_mapping = read_mapping_inverse("./mapping.txt")
    dataset, label, fam_idx, fam_dict, dataset_wl = GNN_script.init_dataset("./databases/examples_samy/BODMAS/01", families, reversed_mapping, [], {}, False)
    print(f"GNN Dataset length: {len(dataset)}")
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
    model = GINJKFlag(test_dataset[0].num_node_features, hidden, num_classes, num_layers, drop_ratio=drop_ratio, residual=residual).to(DEVICE)
    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    
    # FL strategy
    strategy = fl.server.strategy.MKFedAvg(
        fraction_fit=0.2,  # Fraction of available clients used for training at each round
        min_fit_clients=2,  # Minimum number of clients used for training at each round (override `fraction_fit`)
        min_available_clients=2,  # Minimum number of all available clients to be considered
        evaluate_fn=get_evaluate_enc_fn(model, test_dataset, id),  # Evaluation function used by the server 
        evaluate_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(model, test_dataset, id,["accuracy"]),
        fit_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(model, test_dataset, id,["accuracy"]),
        on_fit_config_fn=fit_config,  # Called before every round
        on_evaluate_config_fn=evaluate_config,  # Called before evaluation rounds
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    fl.server.start_server(
        length=len(np.hstack(np.array([val.cpu().numpy().flatten() for _, val in model.state_dict().items()],dtype=object),dtype=object)),
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=nrounds),
        strategy=strategy,
        certificates=(
        Path("./FL/.cache/certificates/ca.crt").read_bytes(),
        Path("./FL/.cache/certificates/server.pem").read_bytes(),
        Path("./FL/.cache/certificates/server.key").read_bytes(),
    )

    )