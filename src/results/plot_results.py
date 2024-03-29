import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse

def sort_xp(filepath,n_xp):
    with open(filepath, "r") as f:
        last_line = f.readlines()[-n_xp-1].strip().split()
        print(last_line)
    
    #xp=pd.read_csv(filepath,index_col=False,delim_whitespace=True,header=None)
    #print(xp.head())
    d = {"folder":last_line[0][:-2]}
    cl=[]
    for r in last_line[1:]:
        if r.find("server")!=-1:
        #if r[24]=="s":
            d["server"]=r
        else:
            cl.append(r)
    d["clients"]=sorted(cl)

    return d

def read_xp(filepath,n_xp,reverse=True):
    with open(filepath, "r") as f:
        if reverse:
            last_line = f.readlines()[-n_xp-1].strip().split()
        else:
            last_line = f.readlines()[n_xp].strip().split()
        #last_line = f.readlines()[-n_xp-1].strip().split()
        print(last_line)
    #xp=pd.read_csv(filepath,index_col=False,delim_whitespace=True,header=None)
    #print(xp.head())
    d = {"folder":last_line[0][:-2]}
    cl=[]
    for r in last_line[1:]:
        if r.find("server")!=-1:
        #if r[24]=="s":
            d["server"]=r
        else:
            cl.append(r)
    d["clients"]=sorted(cl)

    return d

def read_client_results(filepath):
    clients=[]
    for file in filepath:#glob.glob(os.path.join(filepath,"client*.csv")):
        print(file)
        clients.append(pd.read_csv(file,index_col=False).drop(columns=["model"]))
    return clients

def read_server_results(filepath):
    #for file in glob.glob(os.path.join(filepath,"server*.csv")):
    #    return pd.read_csv(file,index_col=False,header=0).drop(columns=["model"]).set_index("metric")
    return [pd.read_csv(filepath,index_col=False,header=0).drop(columns=["model"]).set_index("metric")]
def plot_metric_client(clients_results,metric,path=None):
    l=[]
    print("plot_metric_client", type(clients_results),type(clients_results[0]))
    for i,client in enumerate(clients_results):
        y=client[metric]
        x=[i for i in range(len(y))]
        plt.plot(x,y)
        plt.xticks(x)
        l.append(f"Client {i+1}")
    plt.xlabel("Round [/]")
    if metric.find("time")!=-1:
        plt.ylabel(f"{metric} [s]")
    else:
        plt.ylabel(f"{metric} [/]")
    plt.title(f"Client {metric} for each round")
    plt.legend(l)
    if path is not None:
        plt.savefig(path)
    plt.show()

def plot_metric_server(server_results,metric,types,path=None):
    l=[]
    for t in types:
        m=metric+f"_{t}"
        y=server_results.loc[m]
        x=[i for i in range(len(y))]
        plt.plot(x,y)
        plt.xticks(x)
        l.append(m)
    plt.xlabel("Round [/]")
    if metric=="test_time":
        plt.ylabel(f"{metric} [s]")
    else:
        plt.ylabel(f"{metric} [/]")
    plt.title(f"Server {metric} for each round")
    plt.legend(l)
    if path is not None:
        plt.savefig(path)
    plt.show()

def mean(data):
    l=data[0]#np.array([data[0][i].to_numpy() for i in range(len(data[0]))])
    print("before mean",l[0])
    for d in data[1:]:
        for i in range(len(d)):
            l[i]+=d[i]
    for i in range(len(l)):
        l[i]=l[i]/len(data)
    print("mean",l[0])
    return l

parser = argparse.ArgumentParser()
parser.add_argument("xp_path", help="filepath to the xp file",type=str)
parser.add_argument("--ni", help="initial line number to read",type=int)
parser.add_argument("--nf", help="final line number to read",type=int)
parser.add_argument("--step", help="step to read",type=int)
args = parser.parse_args()
xp_path = args.xp_path #"./results/xp.txt" or ./results/xp_wo.txt
ni=args.ni
nf=args.nf
reverse=False
step=args.step
if ni is None or nf is None:
    ni=5
    nf=0
    step=-1
    reverse=True

folders = ["./results/20240320-1433","./results/20240320-1440","./results/20240320-1445","./results/20240320-1500","./results/20240320-1546"]
xps=[]
for i in range(ni,nf,step):
    xps.append(read_xp(xp_path,i,reverse))
xp=read_xp("./results/xp.txt",1)
print(xp)
#clients_results=read_client_results([file for file in glob.glob(os.path.join("./results/20240320-1440","client*.csv"))])
#plot_metric_client(clients_results,"Accuracy")#,"./results/plot/c_acc.png")

clients = []
servers = []
print(xps)
for f in xps:
    clients.append(read_client_results(f["clients"]))
    servers.append(read_server_results(f["server"]))

clients_mean=mean(clients)
print("clients 0 0", type(clients_mean))
metrics_clients =["Accuracy","Precision","Recall","F1 score","Balanced Accuracy","Loss","Train time","Test time"]
for m in metrics_clients:
    plot_metric_client(clients_mean,m)#,"./results/plot/acc.png"

server_mean = mean(servers)#"#()"./results/20240321-1001")
print("servers 0 0", type(server_mean[0]))
#print(servers[0][0].head(15))
#print(server_mean.head(15))

types = ["df","d","c"]
metrics_server =["accuracy","precision","recall","f1","balanced_accuracy","loss","test_time"]
for m in metrics_server:
    plot_metric_server(server_mean[0],m,types)




