import argparse

def parse_arg_server():
    # Parse command line argument of server
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
        choices=range(1, 100),
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
        default = "",
        type=str,
        required=False,
        help="Specifies the path for the dataset",
    )
    parser.add_argument(
        "--noce",
        action="store_false",
        help="Specifies if there is contribution evaluation or not",
    )

    parser.add_argument(
        "--methodo",
        default = "",
        type=str,
        required=False,
        help="Specifies the methodology used to deal with client that have low SV"
    )
    parser.add_argument(
        "--threshold",
        default = -1.0,
        type=float,
        required=False,
        help="Specifies the threshold to delete clients"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )

    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )
    
    args = parser.parse_args()
    n_clients = args.nclients
    id = n_clients
    nrounds = args.nrounds
    dataset_name = args.dataset
    methodo = args.methodo
    threshold = args.threshold
    filename = args.filepath
    ce=args.noce
    model_type = args.model
    model_path = args.modelpath

    return n_clients, id, nrounds, dataset_name, methodo, threshold, filename, ce, model_type, model_path

def parse_arg_client():
    # Parse command line argument of client
    parser = argparse.ArgumentParser(description="Flower")    
    parser.add_argument(
        "--partition",
        type=int,
        default=0,
        choices=range(0, 10),
        required=False,
        help="Specifies the id of the client. \
        Picks partition 0 by default",
    )
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
        "--filepath",
        type=str,
        required=False,
        default="./results",
        help="Specifies the path for storing results"
    )
    parser.add_argument(
        "--dataset",
        default = "",
        type=str,
        required=False,
        help="Specifies the path for the dataset"
    )
    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )

    args = parser.parse_args()
    n_clients = args.nclients
    id = args.partition
    filename = args.filepath
    dataset_name = args.dataset
    model_path = args.modelpath
    model_type = args.model

    return n_clients, id, filename, dataset_name, model_path, model_type

def parse_arg_ce():
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
    parser.add_argument(
        "--modelpath",
        type=str,
        required=False,
        help="Specifies the path for the model"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Specifies the model used"
    )
    args = parser.parse_args()
    n_clients = args.nclients
    dataset_name = args.dataset
    id = n_clients
    enc = args.enc
    filename = args.filepath
    model= args.model
    model_path = args.modelpath

    return n_clients, id, filename,dataset_name,model,model_path,enc