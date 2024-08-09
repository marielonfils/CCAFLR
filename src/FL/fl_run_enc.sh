#!/bin/bash
declare -i nclients="9"
declare -i nrounds="60"
declare -i ntimes="5"
filepath="./results_images/loop3"
filename="./results_images/xp.txt"
filepre="./results_images/loop3/xp_101_"
fileext=".txt"
dataset="images"#"split_scdg1"
model="images"


#for ((l=10; l<11; l++)); do
for ((k=2; k<nclients; k++)); do
    for ((j=0; j<ntimes;j++)); do
        echo "Starting server $j"
        pids=()
        current_date_time="`date +%Y%m%d-%H%M%S` "
        f="${filepre}${k}_${nrounds}${fileext}"
        echo -n $current_date_time >> $f
        echo $f
        python ./FL/fl_server_enc.py --nclients=${k} --nrounds=${nrounds} --filepath=${filepath} --dataset=${dataset} --noce --model=${model}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
        pids+=($!)
        sleep 30
    
        for ((i=0; i<$k; i++)); do
            echo "Starting client $i"
            python ./FL/fl_client_enc.py --nclients=${k} --partition=${i} --filepath=${filepath} --dataset=${dataset} --model=${model}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
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