#!/bin/bash
declare -i nclients="8"
declare -i nrounds="5"
filepath="./results"
dataset="split_scdg1"
methodo=""
threshold="0.0"

echo "Starting server"
python3 ./FL/fl_server_enc.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold}&
sleep 9  # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python3 ./FL/fl_ce_server.py --enc --nclients=${nclients} --dataset=${dataset}&
sleep 9

for ((i=0; i<nclients-1; i++)); do
    echo "Starting client $i"
    python3 ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset}&
done
sleep 2

echo "Starting client 3"
python ./FL/fl_client_enc_random.py --nclients=${nclients} --partition=4 --filepath=${filepath} --dataset=${dataset}&
    
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
