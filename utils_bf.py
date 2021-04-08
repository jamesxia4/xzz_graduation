import torch 
from torch import nn
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
def graph_generator(features):
    # features=features.cpu().detach().numpy()
    thea=0.4
    batch,r,c=features.shape
    adj=torch.zeros((batch,r,r))
    cosine=np.zeros((r,r))
    for i in range(batch):
        adj_batch=cosine_similarity(features[i,:,:])
        adj_mean=np.mean(adj_batch,axis=0)
        adj_mean_m=np.repeat(adj_mean,r,axis=0).reshape((r,r))
        for m in range(r):
            for n in range(r):
                if adj_batch[m][n]>=adj_mean_m[m][n]:
                    cosine[m][n]=1
                elif thea<adj_batch[m][n]<adj_mean_m[m][n]:
                    cosine[m][n]=0.5
                else:
                    cosine[m][n]=0
        adj_batch_=normalize(cosine)       
        adj[i]=torch.from_numpy(adj_batch_)
    return adj
def normalize(mat):
    mat=mat+np.eye(mat.shape[0])
    rowsum = np.array(mat.sum(1))
    r_inv = np.power(rowsum, -0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    D_max=np.diag(r_inv)
    mat=np.matmul(np.matmul(D_max,mat),D_max)
    return mat

if __name__ == "__main__":
    pass
    A=torch.randn(1,2,3) 
    print(graph_generator(A))
