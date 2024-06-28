import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from all_in import *
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
def make_quaternion_mul(kernel):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    dim = kernel.size(1)//4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    assert kernel.size(1) == hamilton.size(1)
    return hamilton
class QGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout, quaternion_ff=True, act=F.relu):
        super(QGNNLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #
        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(self.in_features, self.out_features))
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj, double_type_used_in_graph=False):

        x = self.dropout(input) # Current Pytorch 1.5.0 doesn't support Dropout for sparse matrix

        if self.quaternion_ff:
            hamilton = make_quaternion_mul(self.weight)
            if double_type_used_in_graph:  # to deal with scalar type between node and graph classification tasks
                hamilton = hamilton.double()

            support = torch.matmul(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.matmul(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.matmul(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)

class Graph_channel(torch.nn.Module):
    def __init__(self,channel=16,ratio=4):
        super(Graph_channel, self).__init__()
        self.mlp1=nn.Linear(channel,7)
        self.mlp12 = nn.Linear(channel,7)
        self.mlp2=nn.Linear(14,channel)
        # self.mlp22=nn.Linear(14,channel//2)
    def forward(self, x):
        x=torch.mean(x,dim=[0,1])
        x2=self.mlp1(x)
        t=self.mlp12(x)
        # t=F.relu(t)
        x = torch.cat([x2* torch.cos(t), x2 * torch.sin(t)])
        x = F.relu(x)
        x = self.mlp2(x)
        x=F.sigmoid(x)
        return x

'''Quaternion graph neural network! 2-layer Q4GNN!'''
class QGNN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout=0.5,freeze_epoch=200):
        super(QGNN, self).__init__()
        self.menk1= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)
        self.menk2= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)
        self.menk3= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)


        self.hid=nhid
        self.q4gnn0 = QGNNLayer(nfeat , nhid, dropout=dropout)
        self.q4gnn1 = QGNNLayer(nfeat , nhid, dropout=dropout)
        self.q4gnn2 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn3 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn4 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn5 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.out = QGNNLayer(nhid, nhid, dropout=dropout, quaternion_ff=False, act=lambda x:x)
        self.graph_channel=Graph_channel(nhid)
        self.prediction = nn.Linear(nhid, nclass)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, adj,mask,now_epoch):

        y0=self.q4gnn0(x,adj)
        # y0=self.fnet(y0)
        x = self.q4gnn1(x, adj)
        x1,x2,x3,x4=torch.split(x,self.hid//4,dim=2)

        y1=self.q4gnn2(x1,adj,now_epoch)
        y12=self.menk1(y1,adj)
        x2=y12*x2
        y2=self.q4gnn3(x2,adj,now_epoch)
        y22=self.menk1(y2,adj)
        x3=y22*x3
        y3 = self.q4gnn4(x3, adj,now_epoch)
        y32=self.menk1(y3,adj)
        x4=y32*x4
        y4 = self.q4gnn5(x4, adj,now_epoch)

        y=torch.cat([y1,y2,y3,y4],dim=2)
        y_channel=self.graph_channel(y)
        y = y*y_channel+y0
        y = self.out(y, adj)
        x = y * mask
        # for i in range(10):
        #     print(y.shape)
        graph_embeddings = torch.sum(x, 1) * torch.amax(x, 1)
        graph_embeddings = self.dropout(graph_embeddings)
        prediction_scores = self.prediction(graph_embeddings)
        return prediction_scores

