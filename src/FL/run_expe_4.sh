#!/bin/bash
declare -i nclients="8"
declare -i nrounds="20"
filepath="./results"
dataset="split_scdg1"
methodo=""
threshold="0.0"
model="models/model_server_30.pt"

echo "Starting server"
python3 ./FL/fl_server.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold}&
sleep 60 # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python3 ./FL/fl_ce_server.py --nclients=${nclients} --filepath=${filepath} --dataset=${dataset}&

for ((i=0; i<nclients-1; i++)); do
    echo "Starting client $i"
    python3 ./FL/fl_client.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
done

echo "Starting client random"
python3 ./FL/fl_client_random.py --nclients=${nclients} --partition=7 --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
    
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
