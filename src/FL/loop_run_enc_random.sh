#!/bin/bash
declare -i nclients="8"
declare -i nrounds="20"
filepath="./results"
dataset="split_scdg1"
thresholds=('-0.1' '-0.05' '-0.01')
methodos=('delete' 'delete_one')

for methodo in "${methodos[@]}"
do
  for threshold in "${thresholds[@]}"
  do
    for i in 1 2 3 4 5
    do
    echo "Starting server"
    python3 ./FL/fl_server_enc.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold}&
    sleep 60  # Sleep for 3s to give the server enough time to start

    echo "Starting CE server"
    python3 ./FL/fl_ce_server.py --enc --nclients=${nclients} --filepath=${filepath} --dataset=${dataset}&
    sleep 3
    
    for ((i=0; i<nclients-1; i++)); do
        echo "Starting client $i"
        python3 ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset}&
        sleep 3
    done
    
    sleep 3
    echo "Starting client random"
    python ./FL/fl_client_enc_random.py --nclients=${nclients} --partition=7 --filepath=${filepath} --dataset=${dataset}&

    # This will allow you to use CTRL+C to stop all background processes
    trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
    # Wait for all background processes to complete
    wait
    done
  done
done
