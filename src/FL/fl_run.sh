#!/bin/bash
declare -i nclients="2"
declare -i nrounds="2"
declare -i ntimes="1"
filepath="./results/loop4"
filename="./results/xp.txt"
filepre="./results/loop4/xp_101_noenc_noce_"
fileext=".txt"
dataset="split_scdg1"



for ((j=0; j<ntimes;j++)); do
    echo "Starting server $j"
    pids=()
    current_date_time="`date +%Y%m%d-%H%M%S` "
    f="${filepre}${k}_${nrounds}${fileext}"
    echo -n $current_date_time >> $f
    echo $f
    python ./FL/fl_server.py --nclients=${nclients} --nrounds=${nrounds} --filepath=${filepath} --dataset=${dataset} --noce| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
    pids+=($!)
    sleep 10

    for ((i=0; i<nclients; i++)); do
        echo "Starting client $i"
        python ./FL/fl_client.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} | awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
        pids+=($!)
    done

    for pid in ${pids[*]}; do
        echo "Waiting for pid $pid"
        wait $pid
    done
    echo -e "" >> $f
done

# python ./SemaClassifier/classifier/GNN/GNN_script.py --nclients=${nclients} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait
