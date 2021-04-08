import torch
import math
import numpy as np
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.nn.functional as F
class GraphConvolution(Module):
    def __init__(self,in_features,out_features,bias=False):
        super(GraphConvolution,self).__init__()
        self.in_features=in_features
        self.out_features=out_features
        self.weight=Parameter(torch.FloatTensor(in_features,out_features).cuda())
        # nn.init.xavier_uniform_(self.weight.data,gain=1.414)
        if bias:
            self.bias=Parameter(torch.FloatTensor(out_features))
            # nn.init.xavier_uniform(self.bias.data,gain=1.414)
        else:
            self.register_parameter('bias',None)
        self.reset_parameters()
    def reset_parameters(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv,stdv)
    def forward(self,input,adj):
        # support=torch.bmm(input,self.weight)
        # output=torch.bmm(adj,support)
        agg_feas=torch.bmm(adj,input)
        output=torch.einsum('bnd,df->bnf', (agg_feas, self.weight))
        if self.bias is not None:
            output=output+self.bias
            return output
        else:
            return output
    def  __repr__(self):
        return self.__class__.__name__+'('+str(self.in_features)+'->'+str(self.out_features)+')'
# class Weight_fusion(Module):
#     def __init__(self,input_feature):
#         super(Weight_fusion,self).__init__()
#         self.input_feature=input_feature
#         self.weight=Parameter(torch.FloatTensor(input_feature,input_feature))
#         nn.init.xavier_uniform_(self.weight.data)
#         # self.bias=Parameter(torch.FloatTensor(input_feature))
#         # nn.init.xavier_uniform_(self.bias.data)
#     def forward(self,input):
#         out=torch.einsum('nd,bdf->bnf',(self.weight,input))
#         return out
class Weight_fusion(Module):
    def __init__(self,input_feature):
        super(Weight_fusion,self).__init__()
        self.input_feature=input_feature
        self.weight=Parameter(torch.FloatTensor(input_feature,input_feature))
        self.bias=Parameter(torch.FloatTensor(input_feature))
        self.reset_parameters()
    def reset_parameters(self):
        stdv=1./math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv,stdv)
        self.bias.data.uniform_(-stdv,stdv)
    def forward(self,input):
        out=torch.einsum('nd,bdf->bnf',(self.weight,input)) +self.bias
        return out
