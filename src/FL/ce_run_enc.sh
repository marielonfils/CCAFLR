#!/bin/bash
declare -i nclients="5"
declare -i nrounds="8"
filepath="./results"
dataset="scdg1"

echo "Starting server"
python ./FL/fl_server_enc.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath}& # --dataset=${dataset}&
sleep 5  # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python ./FL/fl_ce_server.py --enc --nclients=${nclients}& # --dataset=${dataset}&

for ((i=0; i<nclients-1; i++)); do
    echo "Starting client $i"
    python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath}& # --dataset=${dataset}&
done
sleep 2
echo "Starting client 4"
python ./FL/fl_client_enc_random.py --nclients=${nclients} --partition=4 --filepath=${filepath}& # --dataset=${dataset}&
    
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
