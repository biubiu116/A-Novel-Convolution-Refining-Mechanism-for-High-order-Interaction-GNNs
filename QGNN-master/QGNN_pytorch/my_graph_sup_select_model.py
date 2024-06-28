import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from q4gnn import *
from all_in import *
class Graph_channel(torch.nn.Module):
    def __init__(self,channel=16,ratio=4):
        super(Graph_channel, self).__init__()
        self.mlp1=nn.Linear(channel,7)
        self.mlp12 = nn.Linear(channel,7)
        self.mlp2=nn.Linear(14,channel)
        # self.mlp22=nn.Linear(14,channel//2)
    def forward(self, x):
        x=torch.mean(x,dim=0)
        x2=self.mlp1(x)
        # x2=F.relu(x2)
        t=self.mlp12(x)
        # t=F.relu(t)
        x = torch.cat([x2* torch.cos(t), x2 * torch.sin(t)])
        x = F.relu(x)
        x = self.mlp2(x)
        x=F.sigmoid(x)
        return x

class QGNN(torch.nn.Module):
    def __init__(self, nfeat,  nclass, nhid=128,dropout=0.5,freeze_epoch=200):
        super(QGNN, self).__init__()
        self.menk1= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)
        self.menk2= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)
        self.menk3= QGNNLayer(nhid//4, nhid//4, dropout=dropout, quaternion_ff=False,act=lambda x:x)


        self.hid=nhid
        self.q4gnn0 = QGNNLayer(nfeat , nhid, dropout=dropout)
        self.q4gnn1 = QGNNLayer(nfeat , nhid, dropout=dropout) # prediction layer
        self.q4gnn2 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn3 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn4 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.q4gnn5 = ALL_select(nhid//4, nhid//4, dropout=dropout,freeze_epoch=freeze_epoch)
        self.out = QGNNLayer(nhid, nclass, dropout=dropout, quaternion_ff=False, act=lambda x:x) # prediction layer
        self.graph_channel=Graph_channel(nhid)

    def forward(self, x, adj,now_epoch):
        y0=self.q4gnn0(x,adj)
        x = self.q4gnn1(x, adj)

        x1,x2,x3,x4=torch.split(x,self.hid//4,dim=1)

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

        y=torch.cat([y1,y2,y3,y4],dim=1)
        y_channel=self.graph_channel(y)
        y = y*y_channel+y0
        y = self.out(y, adj)

        return F.log_softmax(y, dim=1)
class SupQGNN(nn.Module):
    def __init__(self, feature_dim_size, hidden_size, num_GNN_layers, num_classes, dropout,freeze_epoch=150):
        super(SupQGNN, self).__init__()
        self.feature_dim_size = feature_dim_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_GNN_layers = num_GNN_layers
        #
        self.q4gnnlayers = torch.nn.ModuleList()
        for layer in range(self.num_GNN_layers):
            if layer == 0:
                self.q4gnnlayers.append(QGNN(self.feature_dim_size, self.hidden_size, dropout=dropout,freeze_epoch=freeze_epoch))
            else:
                self.q4gnnlayers.append(QGNN(self.hidden_size, self.hidden_size, dropout=dropout,freeze_epoch=freeze_epoch))
        #
        self.predictions = torch.nn.ModuleList()
        self.dropouts = torch.nn.ModuleList()
        # self.predictions.append(nn.Linear(feature_dim_size, num_classes)) # For including feature vectors to predict graph labels???
        for _ in range(self.num_GNN_layers):
            self.predictions.append(nn.Linear(self.hidden_size, self.num_classes))
            self.dropouts.append(nn.Dropout(dropout))

    def forward(self, Adj_block, X_concat, graph_pool,now_epoch):
        prediction_scores = 0
        input = X_concat
        for layer in range(self.num_GNN_layers):
            input = self.q4gnnlayers[layer](input.double(), Adj_block,now_epoch)
            #sum pooling
            graph_embeddings = torch.spmm(graph_pool, input.float())
            graph_embeddings = self.dropouts[layer](graph_embeddings)
            # Produce the final scores
            prediction_scores += self.predictions[layer](graph_embeddings)

        return prediction_scores

def label_smoothing(true_labels: torch.Tensor, classes: int, smoothing=0.1):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)
        true_dist.fill_(smoothing / (classes - 1))
        true_dist.scatter_(1, true_labels.data.unsqueeze(1), confidence)
    return true_dist