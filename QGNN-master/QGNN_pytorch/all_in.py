from q4gnn import *
from q4gnn8 import *
from q4gnn16 import *
from q4gnn32 import *
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from AFDCal import _energy
class ALL_select(Module):
    def __init__(self, in_features, out_features,dropout=0.5,freeze_epoch=200):
        super(ALL_select, self).__init__()
        self.gnn4=QGNNLayer(in_features, out_features, dropout=dropout)
        self.gnn8=QGNNLayer8(in_features, out_features, dropout=dropout)
        self.gnn16=QGNNLayer16(in_features, out_features, dropout=dropout)
        self.gnn32=QGNNLayer32(in_features, out_features, dropout=dropout)
        self.freeze_epoch=freeze_epoch
        self.final=-1
        # self.weight = Parameter(torch.FloatTensor(1,4))
    def forward(self, x, adj,now_epoch):
        if now_epoch<self.freeze_epoch:
            gnn4=self.gnn4(x,adj)
            gnn8=self.gnn8(x,adj)
            gnn16=self.gnn16(x,adj)
            gnn32=self.gnn32(x,adj)
            all=gnn4+gnn8+gnn16+gnn32
            return all
        elif now_epoch==self.freeze_epoch:
            # gnn_list=[self.gnn4,self.gnn8,self.gnn16,self.gnn32]
            # print(now_epoch)
            gnn4 = self.gnn4(x, adj)
            gnn8 = self.gnn8(x, adj)
            gnn16 = self.gnn16(x, adj)
            gnn32 = self.gnn32(x, adj)
            self.final=select([gnn4,gnn8,gnn16,gnn32])
            all=gnn4+gnn8+gnn16+gnn32
            return all
        else:
            gnn_list=[self.gnn4,self.gnn8,self.gnn16,self.gnn32]
            y=gnn_list[self.final](x,adj)
            return y


def select(selections):
    max=0
    i=-1
    ret=-1
    for select in selections:
        i+=1
        # weight=select.weight
        # print(weight)
        # select=select.reshape(1,-1)
        ener=_energy.total_energy(select)
        # print(select.shape)
        # print(selections.shape)
        print(ener)
        if ener>max:
            ret=i
            max=ener
        # else:
        #     print(ener)
    if ret==-1:
        print('no')
    return ret
