import torch 
from torch import nn
import numpy as np
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
def enclidDistance(x1,x2,sqrt_flag=False):
    res=torch.sum((x1-x2)**2)
    if sqrt_flag:
        res=torch.sqrt(res)
    return res
def calEuclidDistanceMatrix(X):
    S=torch.zeros(len(X),len(X))
    for i in range(len(X)):
        for j in range((i+1),len(X)):
            S[i][j]=1.0*enclidDistance(X[i],X[j])
            S[j][i]=S[i][j]
    return S
def knn(S,k,sigma=1):
    N=len(S)
    A=torch.zeros(N,N)
    for i in range(N):
        dist_with_index=zip(S[i],range(N))
        dist_with_index=sorted(dist_with_index,key=lambda x:x[0])
        neighbours_id=[dist_with_index[m][1] for m in range(1,k+1)]
        count=0
        for j in neighbours_id:
            # A[i][j]=torch.exp(-S[i][j]/(2*sigma*sigma))
            A[i][j]=1-count/N
            count=count+1
    A=(A+A.t())/2
    return A
              
def normalize(mat):
    print(torch.cuda.is_available())
    print(torch.cuda.current_device())
    mat=mat+torch.eye(mat.shape[0]).cuda()
    rowsum = torch.sum(mat,dim=1)
    r_inv=torch.pow(rowsum,-0.5).flatten()
    r_inv[torch.isinf(r_inv)]=0.
    D_max=torch.diag(r_inv)
    mat=D_max.matmul(mat).matmul(D_max)
    return mat
def batch_normalize(adj):
    batch,_,_=adj.shape
    batch_adj=torch.zeros(adj.shape).cuda()
    for i in range(batch):
        batch_adj[i]=normalize(adj[i])
    return batch_adj
# def graph_generator(features):
#         # features=features.cpu().detach().numpy()
#     batch,r,c=features.shape
#     adj=torch.zeros(batch,r,r)
#     for i in range(batch):
#         S=calEuclidDistanceMatrix(features[i])
#         adj[i]=knn(S,6,1)
#         adj[i]=normalize(adj[i])       
#     return adj
def graph_generator_ft(features):
    batch,r,c=features.shape
    adjf,adjt=torch.zeros((batch,r,r)).cuda(),torch.zeros((batch,r,r)).cuda()
    for i in range(batch):
    
        S=calEuclidDistanceMatrix(features[i])
        adjf[i]=knn(S,3,1)
        for j in range(r):
            for k in range(j):
                adjt[i][j][k]=(1-abs(j-k)/r if (abs(k-j)<3) else 0)
                adjt[i][k][j]=adjt[i][j][k]
    return adjf,adjt
# def time_graph_generator():
#     adj=torch.zeros((1,32,32))
#     for i in range(1):
#         for j in range(32):
#             for k in range(32):
#                  adj[i][j][k]=-abs(i-j)
#         adj[i]=normalize(torch.exp(adj)[i])
#     # adj=torch.from_numpy(adj)
#     return adj
    
# def normalize(mat):
#     # mat=mat+np.eye(mat.shape[0])
#     rowsum = np.array(mat.sum(1))+1e-7              #防止出现0除问题
#     r_inv = np.power(rowsum, -0.5).flatten()
#     r_inv[np.isinf(r_inv)] = 0.
#     D_max=np.diag(r_inv)
#     mat=np.matmul(np.matmul(D_max,mat),D_max)
#     return mat

if __name__ == "__main__":
    pass
    A=torch.ones(2,10,3).cuda()
    print(graph_generator_ft(A))
