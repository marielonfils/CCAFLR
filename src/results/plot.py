import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import numpy as np
import argparse
import torch
from SemaClassifier.classifier.GNN.models.GINEClassifier import GINE

class PlotResults:
    def __init__(self, xp_path, xp_name, ni=None, nf=None, step=None, reverse=False, pred=False):
        print(os.getcwd())
        print(os.path.dirname(__name__))
        self.ni = ni
        self.nf = nf
        self.step = step
        self.reverse = reverse
        if self.ni is None or self.nf is None:
            self.ni = 5
            self.nf = 0
            self.step = -1
            self.reverse = True
        print("REVERSE",self.reverse, self.ni, self.nf)
        self.pred=pred
        
        self.metrics_clients = ["Accuracy", "Precision", "Recall", "F1 score", "Balanced Accuracy", "Loss", "Train time", "Test time"]
        self.types = ["df", "d", "c"]
        self.metrics_server = ["accuracy", "precision", "recall", "f1", "balanced_accuracy", "loss", "test_time"]

        self.xp_data={}
        self.xp_names=[]
        self.xp_path_data={}
        self.clients_data={}
        self.server_data={}
        self.c_labels_data={} 
        self.s_labels_data={} 
        self.c_predictions_data={}
        self.s_predictions_data={}
        self.clients_mean_data={}
        self.server_mean_data={}
        self.c_parameters={}
        self.s_parameters={}

        self.load_xp(xp_path, xp_name, self.ni, self.nf, self.step,self.reverse, self.pred)

    def load_xp(self,xp_path,xp_name,ni=None,nf=None,step=None,reverse=False, pred=False):        

        self.xp_path_data[xp_name]=xp_path
        self.xp_names.append(xp_name)

        #read experiments file
        if ni is None or nf is None or step is None or reverse is None:
            ni=self.ni
            nf=self.nf
            step=self.step
            reverse=self.reverse
        self.xp_data[xp_name] = []
        for i in range(ni, nf, step):
            self.xp_data[xp_name].append(self.read_xp(xp_path, i, reverse))

        #read results
        self.c_labels_data[xp_name] = []
        self.s_labels_data[xp_name] = []
        self.c_predictions_data[xp_name] = []
        self.s_predictions_data[xp_name] = []
        self.clients_data[xp_name] = []
        self.server_data[xp_name] = []
        skipfooter=0
        if pred:
            skipfooter=1
        for f in self.xp_data[xp_name]:
            c, cl, cp = self.read_client_results(f["clients"],skipfooter)            
            self.clients_data[xp_name].append(c)
            self.c_labels_data[xp_name].append(cl)
            self.c_predictions_data[xp_name].append(cp)
            s,sl,sp =self.read_server_results(f["server"],skipfooter)
            self.server_data[xp_name].append(s)
            self.s_labels_data[xp_name].append(sl)
            self.s_predictions_data[xp_name].append(sp)
            
            self.c_parameters[xp_name] = {}
            for i in range(len(f["clients_parms"])): # for each client
                self.c_parameters[xp_name][i]={}
                cg_parms,cl_parms = self.read_client_parameters(f["clients_parms"][i])
                self.c_parameters[xp_name][i]["global"]=cg_parms
                self.c_parameters[xp_name][i]["local"]=cl_parms
            self.s_parameters[xp_name] = {}
            s_parms = self.read_server_parameters(f["server_parms"][0])
            self.s_parameters[xp_name][len(f["clients_parms"])]={"server":s_parms}


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
    
    def plot_server_m_r(self,xp_names,metrics=None, types=None, separate=True):
        if metrics is None:
            metrics = self.metrics_server
        if types is None:
            types = self.types
        l=[]
        for xp_name in xp_names:
            for m in metrics:
                l.extend(self.plot_metric_server(self.server_mean_data[xp_name][0], m, types,xp_name, separate=separate))
        if not separate:
            plt.legend(l)
            plt.show()

    def read_xp(self,filepath,n_xp,reverse=False):
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
        clients_path=""
        server_path=""
        ce_server_path=""
        for r in last_line[1:]:
            if r.find("server")!=-1:
            #if r[24]=="s":
                d["server"]=r
                server_path=r[:-27]
            elif r.find("ce")!=-1:
                d["ce"]=r
                ce_server_path=r[:-27]
            else:
                cl.append(r)
                clients_path=r[:-27]
        client_index = len(cl)
        d["clients"]=sorted(cl)
        d["clients_parms"]=[]
        for i in range(client_index):
            d["clients_parms"].append(clients_path+f"parms_{i}/")
        d["server_parms"]=[server_path+f"parms_{client_index}/"]
        print("client path", clients_path,len(d["clients_parms"]),cl)

        return d
    
    def read_client_parameters(self,filepath):
        g={}
        l={}
        for file in sorted(glob.glob(os.path.join(filepath,"model_global_*.pt"))):
            i=len(filepath+"model_global_")
            e=len(file)-3
            index=file[i:e]
            g[index]=file
            #g.append(file)
        for file in sorted(glob.glob(os.path.join(filepath,"model_local_*.pt"))):
            i=len(filepath+"model_local_")
            e=len(file)-3
            index=file[i:e]
            l[index]=file
            #l.append(file)
        return g,l
    
    def read_server_parameters(self,filepath):
        s={}
        for file in sorted(glob.glob(os.path.join(filepath,"model_server_*.pt"))):
            i=len(filepath+"model_server_")
            e=len(file)-3
            index=file[i:e]
            s[index]=file
            #s.append(file)
        return s

    def read_client_results(self,filepath, pred=0):
        clients=[]
        predictions=[]
        labels=[]
        for file in filepath:#glob.glob(os.path.join(filepath,"client*.csv")):
            print(file)
            if pred == 0:
                clients.append(pd.read_csv(file,index_col=False,skipfooter=pred).drop(columns=["model"]))
            else:
                f=pd.read_csv(file,index_col=False,skipfooter=pred,engine='python')
                predictions.append(f["Predictions"])
                clients.append(f.drop(columns=["model","Predictions"]))
                with open(file, "r") as f:
                    labels.append(list(f.readlines()[-1]))
        return clients, labels, predictions

    def read_server_results(self,filepath, pred=0):
        #for file in glob.glob(os.path.join(filepath,"server*.csv")):
        #    return pd.read_csv(file,index_col=False,header=0).drop(columns=["model"]).set_index("metric")
        labels=[]
        server=[]
        prediction=[]
        if pred ==0:
            server=[pd.read_csv(filepath,index_col=False,header=0,skipfooter=pred).drop(columns=["model"]).set_index("metric")]    
        else:
            f=pd.read_csv(filepath,index_col=False,header=0,skipfooter=pred,engine='python').set_index("metric")
            print("pred",f.loc["predictions_c"])
            prediction.append(f.loc["predictions_c"])
            f2=f.drop("predictions_c")
            server=[f2.drop(columns=["model"]).astype(float)]
            print("SERVER", server, server[0].dtypes)
            with open(filepath, "r") as f:
                labels.append(list(f.readlines()[-1]))
        return server,labels,prediction
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

    def plot_metric_server(self,server_results,metric,types,xp_name,path=None, separate=True):
        l=[]
        for t in types:
            m=metric+f"_{t}"
            y=server_results.loc[m]
            x=[i for i in range(len(y))]
            plt.plot(x,y)
            plt.locator_params(nbins=8)
            #plt.xticks(x)
            l.append(xp_name)#+m)
        plt.xlabel("Round [/]")
        if metric=="test_time":
            plt.ylabel(f"{metric} [s]")
        else:
            plt.ylabel(f"{metric} [/]")
        plt.title(f"{metric} for each round")
        
        if path is not None:
            plt.savefig(path)
        if separate:
            plt.show()
            plt.legend(l)
        return l
    

    def mean(self,data):
        #data = array of x lists for  x experiments
        #list = array of n dataframes for n clients
        # dataframe = pandas dataframe containing the results
        l=data[0]#experiment 0 #np.array([data[0][i].to_numpy() for i in range(len(data[0]))])
        print("before mean",l[0].dtypes, type(l), type(l[0]),len(data), len(l),len(l[0]))
        for d in data[1:]: # for each experiment
            for i in range(len(d)): #for each client
                l[i]+=d[i]
        for i in range(len(l)):
            print("before division", l[i], type(l[i]))
            l[i]=l[i]/len(data)
        print("mean",l[0])
        return l


    def parameters_diff(self,path1,path2):
        print(path1,path2)
        model1 = torch.load(path1)
        model2 = torch.load(path2)
        params_model1 = np.hstack(np.array([val.cpu().numpy().flatten() for _, val in model1.state_dict().items()],dtype=object))
        params_model2 = np.hstack(np.array([val.cpu().numpy().flatten() for _, val in model2.state_dict().items()],dtype=object))
        return [params_model2[i] - params_model1[i] for i in range(len(params_model1))]


    def plot(self,xs,ys,labels,xlabel,ylabel,title):
        for x,y,l in zip(xs,ys,labels):
            plt.plot(x,y,label=l)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.show()