#!/bin/bash
declare -i nclients="9"
declare -i nrounds="20"
declare -i ntimes="1"
filepath="./results/loop3"
filename="./results/xp.txt"
filepre="./results/loop3/xp_101"
fileext=".txt"
dataset="split_scdg1"


#for ((l=10; l<11; l++)); do
for ((k=8; k<nclients; k++)); do
    for ((j=0; j<ntimes;j++)); do
        echo "Starting server $j"
        pids=()
        current_date_time="`date +%Y%m%d-%H%M%S` "
        f="${filepre}${k}_${nrounds}${fileext}"
        echo -n $current_date_time >> $f
        echo $f
        python ./FL/fl_server_enc.py --nclients=${k} --nrounds=${nrounds} --filepath=${filepath} --dataset=${dataset} --noce| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
        pids+=($!)
        sleep 30
    
        echo "Starting CE server"
        python ./FL/fl_ce_server.py --nclients=${k} --filepath=${filepath} --dataset=${dataset} --enc| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f & #--dataset=${dataset}&
        pids+=($!)
    
        for ((i=0; i<$k; i++)); do
            echo "Starting client $i"
            python ./FL/fl_client_enc.py --nclients=${k} --partition=${i} --filepath=${filepath} --dataset=${dataset}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
            pids+=($!)
        done
    
        for pid in ${pids[*]}; do
            echo "Waiting for pid $pid"
            wait $pid
        done
        echo -e "" >> $f
    done    
done
#done



# python ./SemaClassifier/classifier/GNN/GNN_script.py --nclients=${nclients} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait