#!/bin/bash
declare -i nclients="8"
declare -i nrounds="60"
filepath="./results_images"
dataset="images"#"split_scdg1"
methodo=""
threshold="0.0"
model="images"

echo "Starting server"
python3 ./FL/fl_server.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold} --model=${model}&
sleep 60  # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python3 ./FL/fl_ce_server.py --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --model=${model}&

for ((i=0; i<nclients; i++)); do
    echo "Starting client $i"
    python3 ./FL/fl_client.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} --model=${model}&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
