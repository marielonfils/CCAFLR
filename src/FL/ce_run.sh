#!/bin/bash
declare -i nclients="10"
declare -i nrounds="3"

echo "Starting server"
python ./FL/fl_server_enc.py --nclients=${nclients}&
sleep 3  # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python ./FL/fl_ce_server.py --nclients=${nclients}&

for ((i=0; i<nclients; i++)); do
    echo "Starting client $i"
    python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i}&
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
