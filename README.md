# CCAFLR - Centralized Coalitional Active Federated Learning with Reputation

The goal of this project is to implement a secure federated learning framework that combines coalitional federated learning, active learning and reputation evaluation, to be able to deal respectively with curious servers and clients, non balanced datasets, wrong labelling and malicious clients.

For the moment, this repository implements a **secure** federated learning framework with **contribution evaluation**.
- *Client data security* is provided by the [xMK-CKKS homomorphic multi-key encryption scheme](https://arxiv.org/abs/2104.06824) \[1\]. The parameters communicated between the clients and the server are encrypted using the aggregated public key and can only be decrypted by the collaboration of all clients.
- *[Contribution evaluation](https://onlinelibrary.wiley.com/doi/full/10.1002/aaai.12082)* \[2\] is realized by a trusted server computing the *Shapley value* \[3\] of each client update. We implemented three client elimination methodologies based on it :
  - *delete_one*: the client with the Shapley value the most under the threshold is discarded.
  - *delete* : all clients with Shapley values under the threshold are discarded.
  - *set_aside* : all client with Shapley values under the threshold are discarded but are given the opportunity rejoin the process at the next contribution evaluation.


## Table of Contents
- [Installation](#installation)
- [Usage](#usage)


## Installation
1. Clone the repository:
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
```

2. Install dependencies:
```bash
 pip install -r requirements.txt
 ```

## Usage
### Scripts
The **CCAFLR/src/FL** folder contains bash scripts to run easily the federated learning task on a single machine:
 - *fl_run.sh* : without encryption and without contribution evaluation
 - *fl_run_enc.sh* : with encryption but without contribution evaluation
 - *ce_run.sh* : without encryption but with contribution evaluation
 - *ce_run_enc.sh* : with encryption and with contribution evaluation.
### Manually
If you want to run each component manually, you need to run the aggregation server, the contribution evaluation server and the clients. 

To run the secure server, use the following command in **CCAFLR/src/**:
```python
python FL/fl_server_enc.py --nclients [number of clients] --nrounds [number of rounds] --filepath [folder path to store results] --dataset [split_scdg1] [--noce] --methodo [delete_one/delete/set_aside] --threshold [threshold value]
```
The *noce* option specifies that no contribution evaluation should be done. In that case, it is unnecessary to specify the *methodo* and *threshold* options.

When evaluating the contributions, *noce* should not be specified. The *delete_one* methodology discards the worst client having a Shapley value under the threshold. The *delete* methodology discards all clients instead of the worst client. The *set_aside* methodology discards all clients and gives them the opportunity to rejoin the process at the next evaluation round. You also need to run the contribution evaluation server using the following command in **CCAFLR/src/**:
```python
python FL/fl_ce_server.py [--enc] --nclients [number of clients] --filepath [folder path to store results] --dataset [split_scdg1]
```

To run the one client, use the following command in **CCAFLR/src/**:
```python
python FL/fl_client_enc.py --nclients [number of clients] --partition [id of the client (0 to number of clients -1)] --filepath [folder path to store results] --dataset [split_scdg1]
```

The *dataset* option specifies the dataset at *src/databases* with the 70% of the data shared equitably among 8 clients and 30% given to the server.
To run the federated learning tast without encryption, replace the filenames of the server and the clients by *fl_server.py* and *fl_client.py*, respectively and remove the *enc** option from the CE server command.




## References
\[1\] Ma, Jing, et al. "Privacy-preserving Federated Learning based on Multi-key Homomorphic Encryption." arXiv preprint arXiv:2104.06824 (2021).
\[2\] Liu, Zelei, et al. "CAreFL: Enhancing smart healthcare with Contribution‐Aware Federated Learning." AI Magazine 44.1 (2023): 4-15.
\[3\] Shapley, Lloyd S. "A value for n-person games." (1953): 307-317.
