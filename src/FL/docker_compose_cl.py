import docker, os

HOSTNAME = os.environ.get("HOSTNAME")
client = docker.from_env()
# get name of current container
name = client.containers.get(HOSTNAME).name
cur_id = name[-1]



# get server IP
serv = client.containers.get('efl-flsrv-1')
ip = serv.attrs['NetworkSettings']['Networks']['efl_fl']['IPAddress']

with open('FL/fl_client_enc.py') as flcl:
    lines = flcl.readlines()

for i in range(len(lines)):
    if '127.0.0.1' in lines[i]:
        lines[i] = lines[i].replace('127.0.0.1', ip)

with open('FL/fl_client_enc.py', "w") as flclw:
    for line in lines:
        flclw.write(line)

print(cur_id)
