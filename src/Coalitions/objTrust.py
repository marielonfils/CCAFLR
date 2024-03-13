from telnetlib import RCP
import numpy as np

#TODO : verify if CPU, RAM and BW have same lengths
def objective_trust(R,H):
	#H:historical values
	#R:measures to evaluate
	delta=0
	etas=np.zeros(len(R))
	for metric in range(len(R)):
		q1 = np.quantile(H[metric],0.25,method="median_unbiased")
		q3 = np.quantile(H[metric],0.75,method="median_unbiased")
		iqr=q3-q1
		LL=q1-iqr*1.5
		print(q1,q3,iqr, LL)
		alpha=0
		rho=0
		for measure in R[metric]:			
			if measure < LL:
				alpha+=measure
				rho+=1
		if rho > 0:
			phi=alpha/rho
			etas[metric]=phi/LL #ATTENTION : line modified (inversed)
			print("rho",alpha,rho,phi,etas[metric])
			delta+=1
	if delta == 0:
		return 1
	else:
		print(etas,delta)
		return np.sum(etas)/delta

CPUA=[0.98]#,0.712,0.711,0.715,0.712]
CPUB=[0.97]
CPUC=[0.96]
CPUD=[0.9]
CPUE=[0.8]
RAM=[0.2]#,0.2,0.28,0.29,0.289]
BW=[0.1]#,0.1,0.1,0.1,0.1]
CPU_H=[0.91,0.92,0.91,0.915,0.92]
RAM_H=[0.25,0.22,0.38,0.49,0.89]
BW_H=[0.1,0.1,0.21,0.1,0.221]
RA=[CPUA,RAM,BW]
RB=[CPUB,RAM,BW]
RC=[CPUC,RAM,BW]
RD=[CPUD,RAM,BW]
RE=[CPUE,RAM,BW]
H=[CPU_H,RAM_H,BW_H]
print("A: ",objective_trust(RA,H),"B: ",objective_trust(RB,H),"C: ",objective_trust(RC,H),"D: ",objective_trust(RD,H),"E: ",objective_trust(RE,H))

def Trust(subj,obj):
	n=len(subj)
	T=0
	for i in range(n):
		T+= (obj+subj[i]) #ATTENTION: removed n* before obj
	T/=(2*n)
	return T
subjD=[0.8,0.9]
objD=1
subjE=[0.8,0.9]
objE=0.89
print(Trust(subjD,objD))
		
	
