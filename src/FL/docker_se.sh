#!/bin/bash
NCLIENTS=$1
NROUNDS=$2
FILEPATH="/results"
DATASET=$3
MODEL=$4
filename="/results/xp.txt"
filepre="xp_101_"
fileext=".txt"

echo "Starting server $j"


python ./FL/docker_compose_se.py

current_date_time="`date +%Y%m%d-%H%M%S` "
f="${FILEPATH}/${filepre}${NROUNDS}_se${fileext}"
echo -n $current_date_time > $f
echo $f
python ./FL/fl_server_enc.py --nclients=${NCLIENTS} --nrounds=${NROUNDS} --filepath=${FILEPATH} --dataset=${DATASET} --noce --model=${MODEL} | awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
pid=($!)
wait $pid
echo -e "" >> $f

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait
