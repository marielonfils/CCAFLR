declare -i nclients="8"
declare -i nrounds="20"
filepath="./results"
dataset="split_scdg1"
model="models/model_server_30.pt"
threshold="-0.01"
methodo="set_aside"

echo "Starting server"
python3 ./FL/fl_server_enc.py --nrounds=${nrounds} --nclients=${nclients} --filepath=${filepath} --dataset=${dataset} --methodo=${methodo} --threshold=${threshold}&
sleep 60 # Sleep for 3s to give the server enough time to start

echo "Starting CE server"
python3 ./FL/fl_ce_server.py --enc --nclients=${nclients} --filepath=${filepath} --dataset=${dataset}&

for ((i=0; i<nclients-3; i++)); do
    echo "Starting client $i"
    python3 ./FL/fl_client_enc.py --nclients=${nclients} --partition=${i} --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
done

echo "Starting client random"
python3 ./FL/fl_client_enc_random.py --nclients=${nclients} --partition=5 --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
    
echo "Starting client random 1"
python3 ./FL/fl_client_enc_random1.py --nclients=${nclients} --partition=6 --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
    
echo "Starting client random 2"
python3 ./FL/fl_client_enc_random2.py --nclients=${nclients} --partition=7 --filepath=${filepath} --dataset=${dataset} --modelpath=${model}&
    
# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait

