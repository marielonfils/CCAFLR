#!/bin/bash
declare -i nclients="3" #"9"
declare -i nrounds="60" #"60"
declare -i ntimes="5" #"5"
declare -i i="0"
filepath="./results_breast/loop5"
filename="./results_breast/loop5/xp.txt"
filepre="./results_breast/loop5/xp_101_"
fileext=".txt"
dataset="breast" #"samy" #"images" #"split_scdg1"
model="mobilenet" #"GINE" #"GINE" #"images"


#for ((l=10; l<11; l++)); do
#for ((k=2; k<nclients; k++)); do
for ((j=0; j<ntimes;j++)); do
    echo "Starting server $j"
    pids=()
    current_date_time="`date +%Y%m%d-%H%M%S` "
    f="${filepre}${nclients}_${nrounds}${fileext}"
    mkdir -p $filepath
    echo -n $current_date_time >> $f
    echo $f
    
    echo "Starting client $i"
    python ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} --model=${model}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f &
    pids+=($!)

    for pid in ${pids[*]}; do
        echo "Waiting for pid $pid"
        wait $pid
    done
    echo -e "" >> $f
done    
#done
#done



# python ./SemaClassifier/classifier/GNN/GNN_script.py --nclients=${nclients} &

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait