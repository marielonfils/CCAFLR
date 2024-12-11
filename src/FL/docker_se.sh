#!/bin/bash
NCLIENTS=$1
NROUNDS=$2
FILEPATH="/results"
DATASET=$3
MODEL=$4
ISENC=$5
SIP=$6
filename="/results/xp.txt"
filepre="xp_101_"
fileext=".txt"

echo "Starting server $j"

if [ $# -eq 5 ]; then
   python ./FL/docker_compose_se.py
else
   echo "IP.3 = $SIP" >> FL/certificates/certificate_docker.conf
   FL/certificates/generate_srv.sh
fi

current_date_time="`date +%Y%m%d-%H%M%S` "
f="${FILEPATH}/${filepre}${NROUNDS}_se${fileext}"
echo -n $current_date_time > $f
echo $f
if [[ "$ISENC" == "true" ]]; then
    python ./FL/fl_server_enc.py --nclients=${NCLIENTS} --nrounds=${NROUNDS} --filepath=${FILEPATH} --dataset=${DATASET} --noce --model=${MODEL} #| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
else
    python ./FL/fl_server.py --nclients=${NCLIENTS} --nrounds=${NROUNDS} --filepath=${FILEPATH} --dataset=${DATASET} --noce --model=${MODEL} #| awk -F"FFFNNN" 'BEGIN { ORS=" " }; !/^$/{print $2}' >> $f
fi
pid=($!)
wait $pid
echo -e "" >> $f

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
echo -e "" >> $filename
# Wait for all background processes to complete
wait
