import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class GAT(nn.Module):
    def __init__(self,in_features, out_features,head_num, alpha, concat=True) -> None:
        super(GAT, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat        
        self.head_num = head_num

        self.every_in_feature = self.in_features//self.head_num
        self.every_out_feature = self.out_features//self.head_num

        self.layers = nn.ModuleList([ GraphAttentionLayer(self.every_in_feature,
                                                          self.every_out_feature,
                                                          self.alpha,
                                                          self.concat) for _ in range(self.head_num)])

    def forward(self, h):
        # h input size: batch_size, N, in_features
        batch_size = h.shape[0]
        agent_num = h.shape[1]
        print("GNN input h shape: ", h.shape)
        # to [batch_size, N, head_num, every_in_features] -> [batch_size, head_num, N, every_in_features]
        h_every_head = h.reshape(batch_size,agent_num,self.head_num,-1).permute(0,2,1,3)
        # [batch_size, agent_num, out_feature]
        return torch.cat([ torch.cat([gat(h_every_head[b_i,head_i,:,:]).unsqueeze(0) for head_i, gat in enumerate(self.layers)],dim=-1) for b_i in range(batch_size)],dim=0)


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """
    def __init__(self, in_features, out_features, alpha, concat=True):
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.empty(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        self.a = nn.Parameter(torch.empty(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, h):
        Wh = torch.mm(h, self.W) # h.shape: (N, in_features), Wh.shape: (N, out_features)
        attention = self._prepare_attentional_mechanism_input(Wh)
        # size: [N,N]
        attention = F.softmax(attention, dim=1)
        # size: [N, out_features] -> [N,out_features]
        h_prime = torch.matmul(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attentional_mechanism_input(self, Wh):
        # Wh.shape (N, out_feature)
        # self.a.shape (2 * out_feature, 1)
        # Wh1&2.shape (N, 1)
        # e.shape (N, N)
        Wh1 = torch.matmul(Wh, self.a[:self.out_features, :])
        Wh2 = torch.matmul(Wh, self.a[self.out_features:, :])
        # broadcast add
        e = Wh1 + Wh2.T
        return self.leakyrelu(e)

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'