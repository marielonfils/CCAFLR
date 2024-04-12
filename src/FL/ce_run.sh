#!/bin/bash
declare -i nclients="3"
declare -i nrounds="3"
filepath="./results"
dataset="scdg1"

echo "Starting server"
python ./FL/fl_server_enc.py --nclients=${nclients} --filepath=${filepath}& #--dataset=${dataset}&
sleep 5  # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python ./FL/fl_ce_server.py --nclients=${nclients}& #--dataset=${dataset}&

for ((i=0; i<nclients; i++)); do
    echo "Starting client $i"
    python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath}& # --dataset=${dataset}&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
