from matplotlib import pyplot as plt
import os
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score , f1_score, balanced_accuracy_score
import pandas as pd
import seaborn as sns

def compute_metrics(y_true, y_pred):
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0.0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0.0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0.0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    return acc, prec, rec, f1, bal_acc

def plot_confusion_matrix(y_true, y_pred, fam_idx, model_name):
    # plot confusion matrix
    if type(y_true[0]) != str:  
        y_true_label = [fam_idx[i] for i in y_true]
        y_pred_label = [fam_idx[i] for i in y_pred]
    else:
        y_true_label = y_true
        y_pred_label = y_pred

    cm = confusion_matrix(y_true_label, y_pred_label, labels=np.unique(fam_idx))
    print(cm)

    df_cm = pd.DataFrame(cm, index = np.unique(fam_idx),
                    columns = np.unique(fam_idx))
    plt.figure(figsize = (10,7))
    heatmap = sns.heatmap(df_cm, annot=True, cmap="Blues", fmt="d",cbar=False)
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=45, ha='right', fontsize=14)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title(f"Confusion matrix for {model_name}")
    plt.savefig(f"confusion_matrix_{model_name}.png")
    # plt.show()

def write_model(filename, model):
    if not os.path.isfile(filename):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename,"w+") as f:
        for v in model:
            f.write(f"{v}: {model[v]}\n")

def write_to_csv(results, file):
    # Write stats and params in csv file
    p = file
    if not os.path.isfile(p):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(p, "w") as f:
            f.write("model,Accuracy,Precision,Recall,F1 score,Balanced Accuracy,Loss,Train time,Test time\n")
    
    with open(p, "a") as f:
        f.write(f"{results[0]},{results[1]},{results[2]},{results[3]},{results[4]},{results[5]},{results[6]},{results[7]},{results[8]}\n")
def write_history_to_csv(hist,model, nrounds, file):
    if not os.path.isfile(file):
        os.makedirs(os.path.dirname(file), exist_ok=True)
        with open(file, "w") as f:
            t="model,metric"
            for r in range(nrounds+1):
                t+=f",round{r}"
            f.write(t+"\n")
    with open(file,"a") as f:
        for m in hist.metrics_distributed_fit:
            t=f"{model.__class__.__name__},{m}_df, 0.0,"
            for v in hist.metrics_distributed_fit[m]:
                t+=f"{v[1]},"
            f.write(t+"\n")
        for m in hist.metrics_distributed:
            t=f"{model.__class__.__name__},{m}_d,"
            for v in hist.metrics_distributed[m]:
                t+=f"{v[1]},"
            f.write(t+"\n")
        for m in hist.metrics_centralized:
            t=f"{model.__class__.__name__},{m}_c,"
            for v in hist.metrics_centralized[m]:
                t+=f"{v[1]},"
            f.write(t+"\n")
    

def write_stats_to_csv(results, clf_model):
    # Write stats and params in csv file
    if not os.path.isfile(f"vec_stats_cv_{clf_model}.csv"):
        with open(f"vec_stats_cv_{clf_model}.csv", "w") as f:
            f.write("model,acc,prec,rec,f1,bal_acc,loss,hidden,layers,lr,batch_size,flag,step_size,m,train_time,test_time\n")
    
    with open(f"vec_stats_cv_{clf_model}.csv", "a") as f:
        f.write(f"{clf_model},{results['final_acc']},{results['final_prec']},{results['final_rec']},{results['final_f1']},{results['final_bal_acc']},{results['final_loss']},{results['best_params']['hidden']},{results['best_params']['layers']},{results['best_params']['lr']},{results['best_params']['batch_size']},{results['best_params']['flag']},{results['best_params']['step_size']},{results['best_params']['m']},{results['training_time']},{results['testing_time']}\n")

def write_stats_to_tmp_csv(results, clf_model):
    # Write stats and params in csv file
    if not os.path.isfile(f"avg_vec_stats_cv_{clf_model}.csv"):
        with open(f"avg_vec_stats_cv_{clf_model}.csv", "w") as f:
            f.write("model,acc,prec,rec,f1,bal_acc,loss,hidden,layers,lr,batch_size,flag,step_size,m,train_time,test_time\n")
    
    with open(f"avg_vec_stats_cv_{clf_model}.csv", "a") as f:
        f.write(f"{clf_model},{results['acc']},{results['prec']},{results['rec']},{results['f1']},{results['bal_acc']},{results['loss']},{results['hidden']},{results['layers']},{results['lr']},{results['batch_size']},{results['flag']},{results['step_size']},{results['m']},{results['training_time']},{results['testing_time']}\n")

def write_cross_val_stats_to_tmp_csv(results, clf_model, fold):
    # Write stats and params in csv file
    if not os.path.isfile(f"folds_vec_stats_cv_{clf_model}.csv"):
        with open(f"folds_vec_stats_cv_{clf_model}.csv", "w") as f:
            f.write("model,acc,prec,rec,f1,bal_acc,loss,hidden,layers,lr,batch_size,fold,flag,step_size,m,train_time,test_time\n")
    
    with open(f"folds_vec_stats_cv_{clf_model}.csv", "a") as f:
        f.write(f"{clf_model},{results['acc']},{results['prec']},{results['rec']},{results['f1']},{results['bal_acc']},{results['loss']},{results['hidden']},{results['layers']},{results['lr']},{results['batch_size']},fold_{fold},{results['flag']},{results['step_size']},{results['m']},{results['training_time']},{results['testing_time']}\n")
