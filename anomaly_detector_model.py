import torch
from torch import nn
from gcn import GraphConvolution,Weight_fusion
import torch.nn.functional as F
from utils import graph_generator_ft,batch_normalize

class AnomalyDetector(nn.Module):
    def __init__(self,dropout=0.6):
        super(AnomalyDetector, self).__init__()
        self.gc1=GraphConvolution(1024,512)
        self.gc2=GraphConvolution(512,32)
        self.gc3=GraphConvolution(32,1)

        self.weight_1=Weight_fusion(32)
        self.weight_2=Weight_fusion(32)
        self.dropout=dropout
        self.sig=nn.Sigmoid()
        self.atte=nn.Sequential(
                 nn.Linear(1024,512),
                 nn.ReLU(),
                 nn.Linear(512,1),
                 nn.Softmax(dim=1)
            )


    def forward(self, x,adjf,adjt):

       adjf_att=self.sig(self.weight_1(adjf))
       adjt_att=self.sig(self.weight_2(adjt))
       adjf_att=(adjf_att+adjf_att.permute(0,2,1))/2
       adjt_att=(adjt_att+adjt_att.permute(0,2,1))/2
       adj=torch.mul(adjf_att,adjf)+torch.mul(adjt_att,adjt)
       res_adj=adj
       adj=batch_normalize(adj).cuda()
       x1=F.relu(self.gc1(x,adj))
       x1=F.dropout(x1,self.dropout,training=self.training)
       x2=F.relu(self.gc2(x1,adj))
       x2=F.dropout(x2,self.dropout,training=self.training)
       x3=self.sig(self.gc3(x2,adj))
       att=self.atte(x)
       vid=(att*x3).sum(dim=1).sum(dim=1)
       return x3,res_adj,vid

def custom_objective(y_pred, y_true,adj,y_bag):
    # y_pred (batch_size, 32, 1)
    # y_true (batch_size)
    lambdas = 8e-5
    y_true=y_true.float()
    y_bag=torch.clamp(y_bag,min=1e-5,max=1.-1e-5)
    loss=-1*(y_true*torch.log(y_bag)+(1.-y_true)*torch.log(1.-y_bag))

    normal_vids_indices = (y_true == 0).nonzero().flatten()
    anomal_vids_indices = (y_true == 1).nonzero().flatten()
    an_adj=adj[anomal_vids_indices]
    normal_segments_scores = y_pred[normal_vids_indices]  # (batch/2, 32, 1)
    anomal_segments_scores = y_pred[anomal_vids_indices]  # (batch/2, 32, 1)

    # just for reducing the last dimension
    normal_segments_scores = torch.sum(normal_segments_scores, dim=(-1,))  # (batch/2, 32)
    anomal_segments_scores = torch.sum(anomal_segments_scores, dim=(-1,))  # (batch/2, 32)

    # get the max score for each video
    normal_segments_scores_maxes = normal_segments_scores.max(dim=-1)[0]
    anomal_segments_scores_maxes = anomal_segments_scores.max(dim=-1)[0]

    hinge_loss = 1.- anomal_segments_scores_maxes + normal_segments_scores_maxes
    hinge_loss = torch.max(hinge_loss, torch.zeros(hinge_loss.shape[0]).cuda())

    sparsity_loss = anomal_segments_scores.sum(dim=-1)
    sloss1=normal_segments_scores.sum(dim=-1)
    adjspass=an_adj.sum(dim=-1).sum(dim=-1)
    # final_loss =0.1*loss.mean()+ (hinge_loss+lambdas*sparsity_loss).mean()+lambdas*Lap.mean()
    final_loss=(hinge_loss+lambdas*sparsity_loss+lambdas*adjspass).mean()+loss.mean()
    return final_loss


class RegularizedLoss(torch.nn.Module):
    def __init__(self, model, original_objective, lambdas=0.001):
        super(RegularizedLoss, self).__init__()
        self.lambdas = lambdas
        self.model = model
        self.objective = original_objective

    def forward(self, y_pred,y_true,adj,vid):
        # loss
        # Our loss is defined with respect to l2 regularization, as used in the original keras code
        we1_params=torch.cat(tuple([x.view(-1) for x in self.model.weight_1.parameters()]))
        we2_params=torch.cat(tuple([x.view(-1) for x in self.model.weight_2.parameters()]))
        gc1_params = torch.cat(tuple([x.view(-1) for x in self.model.gc1.parameters()]))
        gc2_params = torch.cat(tuple([x.view(-1) for x in self.model.gc2.parameters()]))
        gc3_params = torch.cat(tuple([x.view(-1) for x in self.model.gc3.parameters()]))
        # gc4_params = torch.cat(tuple([x.view(-1) for x in self.model.gc4.parameters()]))
        # gc5_params = torch.cat(tuple([x.view(-1) for x in self.model.gc5.parameters()]))
        # gc6_params = torch.cat(tuple([x.view(-1) for x in self.model.gc6.parameters()]))

        l1_regularization = self.lambdas * torch.norm(gc1_params, p=2)
        l2_regularization = self.lambdas * torch.norm(gc2_params, p=2)
        l3_regularization = self.lambdas * torch.norm(gc3_params, p=2)
        l4_regularization = self.lambdas * torch.norm(we1_params, p=2)
        l5_regularization = self.lambdas * torch.norm(we2_params, p=2)
        # l6_regularization = self.lambdas * torch.norm(gc6_params, p=2)




        return self.objective(y_pred, y_true,adj,vid)
   # + l1_regularization + l2_regularization +l3_regularization+ l4_regularization  + l5_regularization
if __name__ == "__main__":
    pass
    import time
    t0=time.time()
    input=torch.randn(2,32,4096).cuda()
    adj1,adj2=graph_generator_ft(input)
    out=AnomalyDetector().forward(input,adj1,adj2)
    t=time.time()-t0
    print(out.shape,'\n',t)

