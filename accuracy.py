import os
import numpy as np
from sklearn.metrics import confusion_matrix,accuracy_score
def loadtxt(dir):
    A=[]
    with open(dir,"r") as f:
        lines=f.read().splitlines()
        for line in lines:
            data=line.split()
            if "Normal" in data[0]:
                A.append(0)
            else:
                A.append(1)
    return A
A=loadtxt("Test_Annotation.txt")
pre=np.load("v_pre.npy")
pre[pre>=0.5]=1
pre[pre<0.5]=0
accuracy=accuracy_score(A,pre)
print(accuracy)
