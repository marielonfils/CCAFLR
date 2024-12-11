mport sys

sip = sys.argv[1]
isenc = sys.argv[2]

fname = 'FL/fl_client_enc.py' if isenc=='true' else 'FL/fl_client.py'

with open(fname) as flcl:
    lines = flcl.readlines()

for i in range(len(lines)):
    if '127.0.0.1' in lines[i]:
        lines[i] = lines[i].replace('127.0.0.1', sip)

with open(fname, "w") as flclw:
    for line in lines:
        flclw.write(line)
