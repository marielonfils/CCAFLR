<!---# CCAFLR - Centralized Coalitional Active Federated Learning with Reputation --->
# MKFL: practical tool for Secure Federated Learning

MKFL is a privacy-preserving federated learning tool based on the xMK-CKKS encryp
tion scheme.

<!---The goal of this project is to implement a secure federated learning framework that combines coalitional federated learning, active learning and reputation evaluation, to be able to deal respectively with curious servers and clients, non balanced datasets, wrong labelling and malicious clients.

For the moment, this repository implements a **secure** federated learning framework.  --->
<!--- with **contribution evaluation**.--->
- *Client data security* is provided by the [xMK-CKKS homomorphic multi-key encryption scheme](https://arxiv.org/abs/2104.06824) \[1\]. The parameters communicated between the clients and the server are encrypted using the aggregated public key and can only be decrypted by the collaboration of all clients.
<!---- *[Contribution evaluation](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12082)* \[2\] is realized by a trusted server computing the *Shapley value* \[3\] of each client update. We implemented three client elimination methodologies based on it :
  - *delete_one*: the client with the Shapley value the most under the threshold is discarded.
  - *delete* : all clients with Shapley values under the threshold are discarded.
  - *set_aside* : all client with Shapley values under the threshold are discarded but are given the opportunity rejoin the process at the next contribution evaluation. --->


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Docker](#docker)
- [Examples](#examples)


## Installation
Works with **Python 3.10.12**.
<!---1. Clone the repository:
```bash
 git clone git@github.com:marielonfils/CCAFLR.git
```
2. Clone the dependencies repositories and install them:
```bash
git clone git@github.com:marielonfils/flower.git
cd flower
pip install .
git clone git@github.com:marielonfils/TenSEAL.git
cd TenSEAL
pip install. 
``` --->


1. Install the dependencies repositories and install them:

Download *flower* from https://anonymous.4open.science/r/flower-4103/README.md. 
```bash
cd flower
pip install .
```
Download *TenSEAL* from https://anonymous.4open.science/r/TenSEAL-8F2A/README.md.
```bash
cd TenSEAL
pip install .
```


2. Install dependencies:
```bash
 cd CCAFLR
 pip install -r requirements.txt
 pip install pyg_lib torch_geometric torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.4.0+cpu.html
 pip install torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cpu
 ```
 
## Usage
### Scripts
The **CCAFLR/src/FL** folder contains bash scripts to run easily the federated learning task on a single machine:
 - *fl_run.sh* : without encryption and without contribution evaluation
 - *fl_run_enc.sh* : with encryption but without contribution evaluation

Do not forget to replace the address of the server at the end of the *src/FL/fl_client_enc.py* or *src/FL/fl_client.py*.
 <!---- *ce_run.sh* : without encryption but with contribution evaluation
 - *ce_run_enc.sh* : with encryption and with contribution evaluation. --->
### Manually
If you want to run each component manually, you need to run the aggregation server, the contribution evaluation server and the clients. 

To run the secure server, use the following command in **CCAFLR/src/**:
```python
python FL/fl_server_enc.py --nclients [number of clients] --nrounds [number of rounds] --filepath [folder path to store results] --dataset [split_scdg1] [--noce] 
```
<!-----methodo [delete_one/delete/set_aside] --threshold [threshold value]
The *noce* option specifies that no contribution evaluation should be done. In that case, it is unnecessary to specify the *methodo* and *threshold* options.--->


<!--- When evaluating the contributions, *noce* should not be specified. The *delete_one* methodology discards the worst client having a Shapley value under the threshold. The *delete* methodology discards all clients instead of the worst client. The *set_aside* methodology discards all clients and gives them the opportunity to rejoin the process at the next evaluation round. You also need to run the contribution evaluation server using the following command in **CCAFLR/src/**:
# ```python
# python FL/fl_ce_server.py [--enc] --nclients [number of clients] --filepath [folder path to store results] --dataset [split_scdg1]>
```

To run one client, use the following command in **CCAFLR/src/**:
```python
python FL/fl_client_enc.py --nclients [number of clients] --partition [id of the client (0 to number of clients -1)] --filepath [folder path to store results] --dataset [name of the dataset] --model [name of the model] [--datapath] [--split[whether to use the whole dataset or to split it]] 
``` --->


To run the federated learning task without encryption, replace the filenames of the server and the clients by *fl_server.py* and *fl_client.py*, respectively.
 <!--- and remove the *enc** option from the CE server command.--->

Do not forget to replace the address of the server at the end of the *src/FL/fl_client_enc.py* or *src/FL/fl_client.py*. 

### Modify the database used

In the *init_datasets* function of *src/FL/main_utils*:
- add a branch with the name of the dataset
- create a Dataset object and provide it an initialization function
- call init_db on the new Dataset and return it


### Modify the model used

In the *get_model* function of *src/FL/main_utils*:
- add a branch with the name of the model
- initialize your model
- create a Model object and provide your model to it
- return the Model object


## Docker Installation and Usage

Docker image can be created with
```bash
docker build ./ -t ccaflr
```
Note that the build takes about 25-30 minutes.

To run the **secure** federated learning framework (without contribution evaluation) on a single machine docker compose can be used
```bash
docker compose up -d
```

*.env* file contains parameters for docker compose: number of clients, number of rounds, dataset name, model name, path to the dataset, and path to a folder to output results.
Latest execution logs can be checked with
```bash
docker compose logs
```

## Examples
Example datasets are provided in *src/databases*.

The first one is the directory *examples_samy*. It contains graphical representation of malware. The dataset to give as parameter is *example_samy* and the model is *GINE*.

The second dataset is *example_images.zip*. It contains image representations of malware. It should first be unzipped in a folder of the same name. The dataset to give as parameter is *example_images* and the model is *images*.


## References
\[1\] Ma, Jing, et al. "Privacy-preserving Federated Learning based on Multi-key Homomorphic Encryption." arXiv preprint arXiv:2104.06824 (2021).
\[2\] Liu, Zelei, et al. "CAreFL: Enhancing smart healthcare with Contribution‐Aware Federated Learning." AI Magazine 44.1 (2023): 4-15.
\[3\] Shapley, Lloyd S. "A value for n-person games." (1953): 307-317.
