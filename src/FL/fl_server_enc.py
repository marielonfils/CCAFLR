import sys
import os
cwd=os.getcwd()
sys.path.insert(0, cwd)
from FL.CE_client_manager import CEClientManager
sys.path.insert(0, cwd+"/SemaClassifier/classifier/GNN")

from SemaClassifier.classifier.GNN import GNN_script
from SemaClassifier.classifier.Images import ImageClassifier as img
from SemaClassifier.classifier.Breast import breast_classifier as bc
import  SemaClassifier.classifier.GNN.gnn_helpers.metrics_utils as metrics_utils
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE
import torch

import flwr as fl

import main_utils
import parse_args

import time
from typing import Dict, Optional, Tuple
from collections import OrderedDict
from pathlib import Path


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


def get_evaluate_enc_fn( valset,id,y_test, dirname, m):
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
        parameters = reshape_parameters(parameters,[x.cpu().numpy().shape for x in m.model.state_dict().values()])
        params_dict = zip(m.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v.astype('f')) for k, v in params_dict})
        m.model.load_state_dict(state_dict, strict=True)
        if dirname is not None:
            torch.save(m.model,f"{dirname}/model_server_{server_round}.pt")
        test_time, loss, y_pred = m.test(m.model,valset,16,DEVICE,id)
        acc, prec, rec, f1, bal_acc = metrics_utils.compute_metrics(y_test, y_pred)
        #metrics_utils.write_to_csv([str(model.__class__.__name__),acc, prec, rec, f1, bal_acc, loss, 0, 0,0,0], filename)
        GNN_script.cprint(f"Client {id}: Evaluation accuracy & loss, {loss}, {acc}, {prec}, {rec}, {f1}, {bal_acc}", id)
        
        return loss, {"accuracy": acc,"precision": prec,"recall": rec,"f1": f1,"balanced_accuracy": bal_acc,"loss": loss, "test_time": test_time, "train_time":0, "predictions": y_pred}

    return evaluate

def get_aggregate_evaluate_enc_fn(m, valset,id,metrics):
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
    n_clients, id, nrounds, dataset_name, methodo, threshold, filename, ce, model_type, model_path = parse_args.parse_arg_server()
    
    #For storing results
    dirname=None
    if filename is not None:
        timestr1 = time.strftime("%Y%m%d-%H%M%S")
        timestr2 = time.strftime("%Y%m%d-%H%M")
        filename1 = f"{filename}/{timestr2}/model.txt"
        filename2 = f"{filename}/{timestr2}/setup.txt"
        dirname=f"{filename}/{timestr2}/parms_{id}/"
        filename = f"{filename}/{timestr2}/server{id}_{timestr1}.csv"
    print("FFFNNN",filename)
    if not os.path.isdir(dirname):
        os.makedirs(os.path.dirname(dirname), exist_ok=True)
    
    #Dataset Loading
    #Modify database initialization function here. THis function should return :
    #   - trainset : list or pytorch Dataset containing the training data, 
    #   - train labels : list containing the training labels, 
    #   - testset : list or pytorch Dataset containing the test data,
    #   - test labels : list containing the test labels,
    #   - classes : list containing the classes names,
    #   - others : dictionary with other return values
    #init_db = main_utils.init_datasets_split_scdg1
    #init_db = bc.init_datasets_breast
    #Dataset attributes are : trainset, y_train, testset, y_test, classes, others
    #d = main_utils.Dataset(init_db)
    # Modify the call to init_datasets
    d = main_utils.init_datasets(dataset_name, n_clients, id)
    # Modify the call to init_db to match the function signature
    #d.init_db(n_clients,id)
    #d.init_db(id)
    GNN_script.cprint(f"Client {id} : datasets length, {len(d.trainset)}, {len(d.testset)}",id)

    #Model Initialization
    batch_size = 32
    hidden = 64
    num_classes = len(d.classes)
    print("NUM_CLASSES",num_classes)
    num_layers = 2#5
    drop_ratio = 0.5
    residual = False
    #Modify model here
    #model = GINE(hidden, num_classes, num_layers).to(DEVICE)
    #model = bc.MobileNet(0.1,0.7,num_classes=2)
    #mo= main_utils.get_model(model_type, d.classes, d.trainset)
    #m = main_utils.Model(model, GNN_script.train, GNN_script.test)
    #train_function = bc.train
    #test_function = bc.test
    #m = main_utils.Model(model, train_function, test_function)
    #Modify the get_model function in main_utils
    m = main_utils.get_model(model_type, d.classes, d.trainset)
    model_parameters = [val.cpu().numpy() for _, val in m.model.state_dict().items()]


    #Save model and setup information
    if filename is not None:
        dict ={"model":m.model.__class__.__name__,"batch_size":batch_size,"hidden":hidden,"num_classes":num_classes,"num_layers":num_layers,"drop_ratio":drop_ratio,"residual":residual,"device":DEVICE,"n_clients":n_clients,"id":id,"nrounds":nrounds,"filename":filename,"classes":d.classes,"train_dataset":len(d.trainset),"test_dataset":len(d.testset),"labels":str(d.y_test)}
        dict2={**dict,**d.others}
        metrics_utils.write_model(filename1,dict2)
        with open(filename2,"w") as f:
          f.write("n_clients: " + str(n_clients) + "\n")
          f.write("nrounds: " + str(nrounds) + "\n")
          f.write("dataset_name: " + str(dataset_name) + "\n")
          f.write("methodo: " + str(methodo) + "\n")
          f.write("threshold: " + str(threshold) + "\n")
    
    # FL strategy    
    strategy = fl.server.strategy.MKFedAvg(
        fraction_fit=0.2,  # Fraction of available clients used for training at each round
        min_fit_clients=n_clients,#2,  # Minimum number of clients used for training at each round (override `fraction_fit`)
        min_evaluate_clients=n_clients,  # Minimum number of clients used for testing at each round 
        min_available_clients=n_clients,#2,  # Minimum number of all available clients to be considered
        evaluate_fn=get_evaluate_enc_fn( d.testset, id,d.y_test, dirname,m),  # Evaluation function used by the server 
        evaluate_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(m, d.testset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time","train_time"]),
        fit_metrics_aggregation_fn=get_aggregate_evaluate_enc_fn(m, d.testset, id,["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time","train_time"]),
        on_fit_config_fn=fit_config,  # Called before every round
        on_evaluate_config_fn=evaluate_config,  # Called before evaluation rounds
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )
    
    shapes=[x.cpu().numpy().shape for x in m.model.state_dict().values()]
    client_manager = CEClientManager()
    # import pdb; pdb.set_trace()

    # Start Flower server
    hist=fl.server.start_server(
        length=len(np.hstack(np.array([val.cpu().numpy().flatten() for _, val in m.model.state_dict().items()],dtype=object),dtype=object)),
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=nrounds),
        strategy=strategy,
        client_manager=client_manager,
        certificates=(
            Path("./FL/.cache/certificates/ca.crt").read_bytes(),
            Path("./FL/.cache/certificates/server.pem").read_bytes(),
            Path("./FL/.cache/certificates/server.key").read_bytes(),),
        enc=True,
        contribution=ce,
        shape=shapes,
        methodo = methodo,
        threshold = threshold,
    )

    #Write results to file
    if filename is not None:
        metrics_utils.write_history_to_csv(hist,m.model, nrounds, filename)
        with open(filename,'a') as f:
            f.write(str(d.y_test)+"\n")
    return

if __name__ == "__main__":
    main()
