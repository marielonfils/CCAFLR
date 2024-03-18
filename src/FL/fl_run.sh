#!/bin/bash
declare -i nclients="2"
declare -i nrounds="3"

echo "Starting server"
python ./FL/fl_server_enc.py --nclients=${nclients}&
sleep 10  # Sleep for 3s to give the server enough time to start



for ((i=0; i<nclients; i++)); do
    echo "Starting client $i"
    python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i}&
done

# python ./SemaClassifier/classifier/GNN/GNN_script.py --nclients=${nclients} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait