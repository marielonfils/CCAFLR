import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")
#classifiers
import torch
from SemaClassifier.classifier.GNN import GNN_script
import main_utils
import parse_args
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
from SemaClassifier.classifier.Images import ImageClassifier as img
#flower
import flwr as fl
from FL.CE_client_manager import CEClientManager

from pathlib import Path
import numpy as np
from typing import Dict, Optional, Tuple
from collections import OrderedDict



import time

DEVICE: str = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def get_evaluate_fn(m, valset,id,y_test,dirname):
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
        params_dict = zip(m.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        m.model.load_state_dict(state_dict, strict=True)
        if dirname is not None:
            torch.save(m.model,f"{dirname}/model_server_{server_round}.pt")
        
        #test_time, loss, y_pred  = GNN_script.test(model, valset, 32, DEVICE,id)
        test_time, loss, y_pred =m.test(m.model,valset, 16, id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
        #metrics_utils.write_to_csv([str(model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, 0, 0,0,0], filename)
        main_utils.cprint(f"Client {id}: Evaluation accuracy & loss, {loss}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", id)
        
        return loss, {"accuracy": acc,"precision": prec,"recall": rec,"f1": f1,"balanced_accuracy": bal_acc,"loss": loss, "test_time": test_time, "train_time":0, "predictions": y_pred}#str(np.array_str(np.array(y_pred),max_line_width=10**50))}

    return evaluate

def get_aggregate_evaluate_fn(model: torch.nn.Module, valset,id,metrics):
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
        metrics = eval_metrics[0][1].keys()
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
    n_clients, id, nrounds, dataset_name, methodo, threshold, filename, ce, model_type, model_path, datapath, split = parse_args.parse_arg_server()
    dirname=""
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename1 = f"{filename}/{timestr2}_wo/model.txt"
        dirname=f"{filename}/{timestr2}_wo/parms_{id}/"
        filename2 = f"{filename}/{timestr2}_wo/setup.txt"
        filename = f"{filename}/{timestr2}_wo/server{id}_{timestr1}.csv"
    print("FFFNNN",filename)

    if not os.path.isdir(dirname):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)

    #Dataset Loading
    #Modify the init_datasets function in main_utils
    d= main_utils.init_datasets(dataset_name, datapath, split, n_clients, id)
    main_utils.cprint(f"Client {id} : datasets length, {len(d.trainset)}, {len(d.testset)}",id)

    #Model
    m = main_utils.get_model(model_type, d.classes, d.trainset, model_path)
    model_parameters = [val.cpu().numpy() for _, val in m.model.state_dict().items()]
    if filename is not None:
        dict ={"model":m.model.__class__.__name__,"device":DEVICE,"n_clients":n_clients,"id":id,"nrounds":nrounds,"filename":filename,"classes":d.classes,"train_dataset":len(d.trainset),"test_dataset":len(d.testset),"labels":str(d.y_test)}
        dict2={**dict,**m.parameters, **d.others}
        metrics_utils.write_model(filename1,dict2)
        with open(filename2,"w") as f:
          f.write("n_clients: " + str(n_clients) + "\n")
          f.write("nrounds: " + str(nrounds) + "\n")
          f.write("dataset_name: " + str(dataset_name) + "\n")
          f.write("methodo: " + str(methodo) + "\n")
          f.write("threshold: " + str(threshold) + "\n")
    #Modify the get_model function in main_utils
    m = main_utils.get_model(model_type, d.classes, d.trainset)

    
    # FL strategy
    strategy = fl.server.strategy.FedAvg(#fl.server.strategy.FedAvg(
        fraction_fit=0.2,  # Fraction of available clients used for training at each round
        min_fit_clients=n_clients,  # Minimum number of clients used for training at each round (override `fraction_fit`)
        min_evaluate_clients=n_clients,  # Minimum number of clients used for testing at each round
        min_available_clients=n_clients,  # Minimum number of all available clients to be considered
        evaluate_fn=get_evaluate_fn(m, d.testset, id,d.y_test, dirname),  # Evaluation function used by the server without enc
        evaluate_metrics_aggregation_fn=get_aggregate_evaluate_fn(m, d.testset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time","train_time"]),
        fit_metrics_aggregation_fn=get_aggregate_evaluate_fn(m, d.testset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time","train_time"]),
        on_fit_config_fn=fit_config,  # Called before every round
        on_evaluate_config_fn=evaluate_config,  # Called before evaluation rounds
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    client_manager = CEClientManager()
    
    hist=fl.server.start_server(
        length = len(np.hstack(np.array([val.cpu().numpy().flatten() for _, val in m.model.state_dict().items()],dtype=object),dtype=object)),
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=nrounds),
        strategy=strategy,
        client_manager=client_manager,
        certificates=(
        Path("./FL/.cache/certificates/ca.crt").read_bytes(),
        Path("./FL/.cache/certificates/server.pem").read_bytes(),
        Path("./FL/.cache/certificates/server.key").read_bytes(),
        ),
        contribution=ce,
        methodo = methodo,
        threshold = threshold,
    )
    if filename is not None:
        metrics_utils.write_history_to_csv(hist,m.model, nrounds, filename)
        with open(filename,'a') as f:
            f.write(str(d.y_test)+"\n")
    return filename

if __name__ == "__main__":
    main()    
