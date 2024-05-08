#!/bin/bash
declare -i nclients="8"
declare -i nrounds="2"
declare -i ntimes="1"
filepath="./results"
filename="./results/xp.txt"
filename1="./results/xp1.txt"
filename2="./results/xp2.txt"
dataset="split_scdg1"


for ((j=0; j<ntimes;j++)); do
    echo "Starting server $j"
    pids=()
    current_date_time="`date +%Y%m%d-%H%M%S` "
    echo -n $current_date_time >> $filename
    python ./FL/fl_server_enc.py --nclients=${nclients} --nrounds=${nrounds} --filepath=${filepath} --dataset=${dataset} | awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $filename &
    pids+=($!)
    sleep 10

    for ((i=0; i<nclients; i++)); do
        echo "Starting client $i"
        python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} | awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $filename &
        pids+=($!)
    done

    for pid in ${pids[*]}; do
        echo "Waiting for pid $pid"
        wait $pid
    done
    echo -e "" >> $filename
done

echo "\n" >> $filename1

# python ./SemaClassifier/classifier/GNN/GNN_script.py --nclients=${nclients} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait
