#!/bin/bash
declare -i nclients="8"
declare -i nrounds="5"
filepath="./results"
dataset="split_scdg1"
methodo="delete"
threshold="0.0"


echo "Starting server"
python3 ./FL/fl_server.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --noce&
sleep 60  # Sleep for 3s to give the server enough time to start

for ((i=0; i<nclients; i++)); do
    echo "Starting client $i"
    python3 ./FL/fl_client.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset}&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
