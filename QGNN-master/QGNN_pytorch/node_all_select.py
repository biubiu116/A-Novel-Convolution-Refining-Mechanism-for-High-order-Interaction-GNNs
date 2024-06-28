from __future__ import division
from __future__ import print_function
from all_in import *
import time
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
np.random.seed(123)
torch.manual_seed(123)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
from utils_node_cls import *
from q4gnn import *

# Parameters
# ==================================================
parser = ArgumentParser("QGNN", formatter_class=ArgumentDefaultsHelpFormatter, conflict_handler='resolve')
parser.add_argument("--dataset",  help="Name of the dataset.")
parser.add_argument('--epochs', type=int,  help='Number of epochs to train.')
parser.add_argument('--learning_rate', type=float,  help='Initial learning rate.')
parser.add_argument('--weight_decay', type=float, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--hidden_size', type=int)
parser.add_argument('--dropout', type=float,  help='Dropout rate (1 - keep probability).')
parser.add_argument('--fold', type=int,  help='The fold index. 0-9.')
parser.add_argument('--fastmode', action='store_true', default=False, help='Validate during training pass.')
args = parser.parse_args()
adj, features, labels, y_train, y_val, y_test, train_mask, val_mask, test_mask = \
    load_data_new_split(args.dataset, '../splits/' + args.dataset + '_split_0.6_0.2_'+ str(args.fold) + '.npz')
labels = torch.from_numpy(labels).to(device)
labels = torch.where(labels==1)[1]
idx_train = torch.where(torch.from_numpy(train_mask)==True)
idx_val = torch.where(torch.from_numpy(val_mask)==True)
idx_test = torch.where(torch.from_numpy(test_mask)==True)
def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape).to(device)

""" preprocess for feature vectors """
def preprocess_features(features):
    """Row-normalize feature matrix"""
    rowsum = np.array(features.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    features = r_mat_inv.dot(features)
    features = features.todense()
    features = np.tile(features, 4) # A + Ai + Aj + Ak
    return torch.from_numpy(features).to(device)

# Some preprocessing
features = preprocess_features(features)
adj = normalize_adj(adj + sp.eye(adj.shape[0])).tocoo()
adj = sparse_mx_to_torch_sparse_tensor(adj)
def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

class Graph_channel(torch.nn.Module):
    def __init__(self,channel=16,ratio=4):
        super(Graph_channel, self).__init__()
        self.mlp1=nn.Linear(channel,7)
        self.mlp12 = nn.Linear(channel,7)
        self.mlp2=nn.Linear(14,channel)
    def forward(self, x):
        x=torch.mean(x,dim=0)
        x2=self.mlp1(x)
        t=self.mlp12(x)
        x = torch.cat([x2* torch.cos(t), x2 * torch.sin(t)])
        x = F.relu(x)
        x = self.mlp2(x)
        x=F.sigmoid(x)
        return x

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
        self.out = QGNNLayer(nhid, nclass, dropout=dropout, quaternion_ff=False, act=lambda x:x)
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

# Model and optimizer
model = QGNN(nfeat=features.size(1), nhid=args.hidden_size, nclass=y_train.shape[1], dropout=args.dropout,freeze_epoch=freeze_epoch).to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

"""Adapted from https://github.com/tkipf/pygcn/blob/master/pygcn/train.py"""
def train(epoch):
    t = time.time()
    model.train()
    optimizer.zero_grad()
    output = model(features, adj,epoch)
    print('inference time: {:.4f}s'.format(time.time() - t))
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    if not args.fastmode:
        # Evaluate validation set performance separately, deactivates dropout during validation run.
        model.eval()
        output = model(features, adj,epoch)

    loss_val = F.nll_loss(output[idx_val], labels[idx_val])
    acc_val = accuracy(output[idx_val], labels[idx_val])
    print('Epoch: {:04d}'.format(epoch+1),
          'loss_train: {:.4f}'.format(loss_train.item()),
          'acc_train: {:.4f}'.format(acc_train.item()),
          'loss_val: {:.4f}'.format(loss_val.item()),
          'acc_val: {:.4f}'.format(acc_val.item()),
          'training time: {:.4f}s'.format(time.time() - t))

def test():
    model.eval()
    output = model(features, adj,epoch)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])
    print("Test set results:",
          "loss= {:.4f}".format(loss_test.item()),
          "accuracy= {:.4f}".format(acc_test.item()))

# Train model
t_total = time.time()
for epoch in range(args.epochs):
    train(epoch)
print("Optimization Finished!")
print("Total time elapsed: {:.4f}s".format(time.time() - t_total))

# Testing
test()


