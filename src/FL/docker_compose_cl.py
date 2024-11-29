import docker, os, sys

isenc = sys.argv[1]
fname = 'FL/fl_client_enc.py' if isenc else 'FL/fl_client.py'


HOSTNAME = os.environ.get("HOSTNAME")
client = docker.from_env()
# get name of current container
name = client.containers.get(HOSTNAME).name
cur_id = name[-1]

serv = client.containers.get('efl-flsrv-1')
sip = serv.attrs['NetworkSettings']['Networks']['efl_fl']['IPAddress']

with open(fname) as flcl:
    lines = flcl.readlines()

for i in range(len(lines)):
    if '127.0.0.1' in lines[i]:
        lines[i] = lines[i].replace('127.0.0.1', sip)

with open(fname, "w") as flclw:
    for line in lines:
        flclw.write(line)

print(cur_id)
