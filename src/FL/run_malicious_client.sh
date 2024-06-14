#!/bin/bash
declare -i nclients="8"
declare -i nrounds="25"
filepath="./results"
dataset="split_scdg1"
thresholds=('-0.1' '-0.05' '-0.01')
methodos=('delete' 'delete_one' 'set_aside')
model="models/model_server_30.pt"

for methodo in "${methodos[@]}"
do
  for threshold in "${thresholds[@]}"
  do
    echo "Starting server"
    python3 ./FL/fl_server_enc.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold}&
    sleep 60  # Sleep for 3s to give the server enough time to start

    echo "Starting CE server"
    python3 ./FL/fl_ce_server.py --enc --nclients=${nclients} --filepath=${filepath} --dataset=${dataset}&
    
    for ((i=0; i<nclients; i++)); do
        echo "Starting client $i"
        python3 ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
    done
    
    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
  done
done
