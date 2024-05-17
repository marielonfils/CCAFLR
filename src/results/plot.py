import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse

class PlotResults:
    def __init__(self, xp_path, xp_name, ni=None, nf=None, step=None):
        self.ni = ni
        self.nf = nf
        self.step = step
        self.reverse = False
        if self.ni is None or self.nf is None:
            self.ni = 5
            self.nf = 0
            self.step = -1
            self.reverse = True
        
        self.metrics_clients = ["Accuracy", "Precision", "Recall", "F1 score", "Balanced Accuracy", "Loss", "Train time", "Test time"]
        self.types = ["df", "d", "c"]
        self.metrics_server = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "loss", "test_time"]

        self.xp_data={}
        self.xp_names=[]
        self.xp_path_data={}
        self.clients_data={}
        self.server_data={}
        self.c_labels_data={}  
        self.clients_mean_data={}
        self.server_mean_data={}

        self.load_xp(xp_path, xp_name, self.ni, self.nf, self.step)

    def load_xp(self,xp_path,xp_name,ni=None,nf=None,step=None):        

        self.xp_path[xp_name]=xp_path
        self.xp_names.append(xp_name)

        #read experiments file
        self.xp_data[xp_name] = []
        for i in range(self.ni, self.nf, self.step):
            self.xp_data[xp_name].append(self.read_xp(xp_path, i, self.reverse))

        #read results
        self.c_labels_data[xp_name] = []
        self.clients_data[xp_name] = []
        self.server_data[xp_name] = []
        for f in self.xps:
            c, l = self.read_client_results(f["clients"])
            
            self.clients_data[xp_name].append(c)
            self.c_labels_data[xp_name].append(l)
            self.server_data[xp_name].append(self.read_server_results(f["server"])) 

        #compute means
        self.clients_mean_data[xp_name] = self.mean(self.clients_data[xp_name])
        self.server_mean_data[xp_name] = self.mean(self.server_data[xp_name]) 

    def plot_client_m_r(self,xp_name, metrics=None, type=None):
        if metrics is None:
            metrics = self.metrics_clients
        if type is None:
            for m in metrics:
                self.plot_metric_client(self.clients_mean_data[xp_name], m)
        if type == "a":
            for m in metrics:
                self.plot_metric_client_afit(self.clients_mean_data[xp_name], m)
        if type == "b":
            for m in metrics:
                self.plot_metric_client_bfit(self.clients_mean_data[xp_name], m)
    
    def plot_server_m_r(self,xp_name,metrics=None, types=None):
        if metrics is None:
            metrics = self.metrics_server
        if types is None:
            types = self.types
        for m in metrics:
            self.plot_metric_server(self.server_mean_data[xp_name][0], m, types)

    def read_xp(self,filepath,n_xp,reverse=True):
        with open(filepath, "r") as f:
            if reverse:
                last_line = f.readlines()[-n_xp].strip().split()
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
            elif r.find("ce")!=-1:
                d["ce"]=r
            else:
                cl.append(r)
        d["clients"]=sorted(cl)

        return d

    def read_client_results(self,filepath):
        clients=[]
        predictions=[]
        for file in filepath:#glob.glob(os.path.join(filepath,"client*.csv")):
            print(file)
            clients.append(pd.read_csv(file,index_col=False,skipfooter=0).drop(columns=["model"]))
            #with open(file, "r") as f:
            #    predictions.append(list(f.readlines()[-1]))
        return clients, predictions

    def read_server_results(self,filepath):
        #for file in glob.glob(os.path.join(filepath,"server*.csv")):
        #    return pd.read_csv(file,index_col=False,header=0).drop(columns=["model"]).set_index("metric")
        return [pd.read_csv(filepath,index_col=False,header=0).drop(columns=["model"]).set_index("metric")]
    def plot_metric_client(self,clients_results,metric,path=None):
        l=[]
        print("plot_metric_client", type(clients_results),type(clients_results[0]))
        for i,client in enumerate(clients_results):
            y=client[metric]
            x=[i for i in range(len(y))]
            plt.plot(x,y)
            #plt.xticks(x)
            plt.locator_params(nbins=8)
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

    def plot_metric_client_bfit(self,clients_results,metric,path=None):
        l=[]
        print("plot_metric_client", type(clients_results),type(clients_results[0]))
        for i,client in enumerate(clients_results):
            y=client[metric]
            x=[i for i in range(len(y))]
            x=[i for i in range(1,len(x[1::2])+1)]
            if i==0:
                print(x[1::2],y[1::2], type(y[1::2]))
            plt.plot(x,y[1::2])
            plt.locator_params(nbins=8)
            #plt.xticks([i for i in range(1,len(x[1::2])+1)])
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

    def plot_metric_client_afit(self,clients_results,metric,path=None):
        l=[]
        print("plot_metric_client", type(clients_results),type(clients_results[0]))
        for i,client in enumerate(clients_results):
            y=client[metric]
            x=[i for i in range(len(y))]
            x=[i for i in range(1,len(x[::2])+1)]
            plt.plot(x,y[::2])
            plt.locator_params(nbins=8)
            #plt.xticks([i for i in range(1,len(x[1::2])+1)])
            l.append(f"Client {i+1}")
        plt.xlabel("Round [/]")
        if metric.find("time")!=-1:
            plt.ylabel(f"{metric} [s]")
        else:
            plt.ylabel(f"{metric} [/]")
        plt.title(f"Client {metric} for each round")
        plt.legend(l)
        if path is not None:
            plt.savefig(path+metric+".png")
        plt.show()

    def plot_metric_server(self,server_results,metric,types,path=None):
        l=[]
        for t in types:
            m=metric+f"_{t}"
            y=server_results.loc[m]
            x=[i for i in range(len(y))]
            plt.plot(x,y)
            plt.locator_params(nbins=8)
            #plt.xticks(x)
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

    def mean(self,data):
        l=data[0]#np.array([data[0][i].to_numpy() for i in range(len(data[0]))])
        print("before mean",l[0])
        for d in data[1:]:
            for i in range(len(d)):
                l[i]+=d[i]
        for i in range(len(l)):
            l[i]=l[i]/len(data)
        print("mean",l[0])
        return l

