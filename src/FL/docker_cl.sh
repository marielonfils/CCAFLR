#!/bin/bash
NCLIENTS=$1
NROUNDS=$2
FILEPATH="/results"
DATASET=$3
MODEL=$4
ISENC=$5
SERVERIP=$6
filename="/results/xp.txt"
filepre="xp_101_"
fileext=".txt"

#if docker on the same machine
# script to find and replace server IP and outputs number of the current client
PART=$(python ./FL/docker_compose_cl.py $SERVERIP)

echo "Starting client ${PART}"

current_date_time="`date +%Y%m%d-%H%M%S` "
f="${FILEPATH}/${filepre}${NROUNDS}_cl${PART}${fileext}"
echo -n $current_date_time > $f
echo $f
sleep 60
if [[ "$ISENC" == "true" ]]; then
    python ./FL/fl_client_enc.py --nclients=${NCLIENTS} --partition=${PART} --filepath=${FILEPATH} --dataset=${DATASET} --model=${MODEL}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
else
    python ./FL/fl_client.py --nclients=${NCLIENTS} --partition=${PART} --filepath=${FILEPATH} --dataset=${DATASET} --model=${MODEL}| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
fi
wait $pid
echo -e "" >> $f

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait


python FL/fl_client.py --nclients=3 --partition=0 --filepath=results_breast/loop7_60_noenc_84 --model=mobilenet --dataset=breast
