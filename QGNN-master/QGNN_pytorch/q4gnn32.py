import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
kernel_list=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32], [2, -1, 4, -3, 6, -5, -8, 7, 10, -9, -12, 11, -14, 13, 16, -15, 18, -17, -20, 19, -22, 21, 24, -23, -26, 25, 28, -27, 30, -29, -32, 31], [3, -4, -1, 2, 7, 8, -5, -6, 11, 12, -9, -10, -15, -16, 13, 14, 19, 20, -17, -18, -23, -24, 21, 22, -27, -28, 25, 26, 31, 32, -29, -30], [4, 3, -2, -1, 8, -7, 6, -5, 12, -11, 10, -9, -16, 15, -14, 13, 20, -19, 18, -17, -24, 23, -22, 21, -28, 27, -26, 25, 32, -31, 30, -29], [5, -6, -7, -8, -1, 2, 3, 4, 13, 14, 15, 16, -9, -10, -11, -12, 21, 22, 23, 24, -17, -18, -19, -20, -29, -30, -31, -32, 25, 26, 27, 28], [6, 5, -8, 7, -2, -1, -4, 3, 14, -13, 16, -15, 10, -9, 12, -11, 22, -21, 24, -23, 18, -17, 20, -19, -30, 29, -32, 31, -26, 25, -28, 27], [7, 8, 5, -6, -3, 4, -1, -2, 15, -16, -13, 14, 11, -12, -9, 10, 23, -24, -21, 22, 19, -20, -17, 18, -31, 32, 29, -30, -27, 28, 25, -26], [8, -7, 6, 5, -4, -3, 2, -1, 16, 15, -14, -13, 12, 11, -10, -9, 24, 23, -22, -21, 20, 19, -18, -17, -32, -31, 30, 29, -28, -27, 26, 25], [9, -10, -11, -12, -13, -14, -15, -16, -1, 2, 3, 4, 5, 6, 7, 8, 25, 26, 27, 28, 29, 30, 31, 32, -17, -18, -19, -20, -21, -22, -23, -24], [10, 9, -12, 11, -14, 13, 16, -15, -2, -1, -4, 3, -6, 5, 8, -7, 26, -25, 28, -27, 30, -29, -32, 31, 18, -17, 20, -19, 22, -21, -24, 23], [11, 12, 9, -10, -15, -16, 13, 14, -3, 4, -1, -2, -7, -8, 5, 6, 27, -28, -25, 26, 31, 32, -29, -30, 19, -20, -17, 18, 23, 24, -21, -22], [12, -11, 10, 9, -16, 15, -14, 13, -4, -3, 2, -1, -8, 7, -6, 5, 28, 27, -26, -25, 32, -31, 30, -29, 20, 19, -18, -17, 24, -23, 22, -21], [13, 14, 15, 16, 9, -10, -11, -12, -5, 6, 7, 8, -1, -2, -3, -4, 29, -30, -31, -32, -25, 26, 27, 28, 21, -22, -23, -24, -17, 18, 19, 20], [14, -13, 16, -15, 10, 9, 12, -11, -6, -5, 8, -7, 2, -1, 4, -3, 30, 29, -32, 31, -26, -25, -28, 27, 22, 21, -24, 23, -18, -17, -20, 19], [15, -16, -13, 14, 11, -12, 9, 10, -7, -8, -5, 6, 3, -4, -1, 2, 31, 32, 29, -30, -27, 28, -25, -26, 23, 24, 21, -22, -19, 20, -17, -18], [16, 15, -14, -13, 12, 11, -10, 9, -8, 7, -6, -5, 4, 3, -2, -1, 32, -31, 30, 29, -28, -27, 26, -25, 24, -23, 22, 21, -20, -19, 18, -17], [17, -18, -19, -20, -21, -22, -23, -24, -25, -26, -27, -28, -29, -30, -31, -32, -1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], [18, 17, -20, 19, -22, 21, 24, -23, -26, 25, 28, -27, 30, -29, -32, 31, -2, -1, -4, 3, -6, 5, 8, -7, -10, 9, 12, -11, 14, -13, -16, 15], [19, 20, 17, -18, -23, -24, 21, 22, -27, -28, 25, 26, 31, 32, -29, -30, -3, 4, -1, -2, -7, -8, 5, 6, -11, -12, 9, 10, 15, 16, -13, -14], [20, -19, 18, 17, -24, 23, -22, 21, -28, 27, -26, 25, 32, -31, 30, -29, -4, -3, 2, -1, -8, 7, -6, 5, -12, 11, -10, 9, 16, -15, 14, -13], [21, 22, 23, 24, 17, -18, -19, -20, -29, -30, -31, -32, 25, 26, 27, 28, -5, 6, 7, 8, -1, -2, -3, -4, -13, -14, -15, -16, 9, 10, 11, 12], [22, -21, 24, -23, 18, 17, 20, -19, -30, 29, -32, 31, -26, 25, -28, 27, -6, -5, 8, -7, 2, -1, 4, -3, -14, 13, -16, 15, -10, 9, -12, 11], [23, -24, -21, 22, 19, -20, 17, 18, -31, 32, 29, -30, -27, 28, 25, -26, -7, -8, -5, 6, 3, -4, -1, 2, -15, 16, 13, -14, -11, 12, 9, -10], [24, 23, -22, -21, 20, 19, -18, 17, -32, -31, 30, 29, -28, -27, 26, 25, -8, 7, -6, -5, 4, 3, -2, -1, -16, -15, 14, 13, -12, -11, 10, 9], [25, 26, 27, 28, 29, 30, 31, 32, 17, -18, -19, -20, -21, -22, -23, -24, -9, 10, 11, 12, 13, 14, 15, 16, -1, -2, -3, -4, -5, -6, -7, -8], [26, -25, 28, -27, 30, -29, -32, 31, 18, 17, 20, -19, 22, -21, -24, 23, -10, -9, 12, -11, 14, -13, -16, 15, 2, -1, 4, -3, 6, -5, -8, 7], [27, -28, -25, 26, 31, 32, -29, -30, 19, -20, 17, 18, 23, 24, -21, -22, -11, -12, -9, 10, 15, 16, -13, -14, 3, -4, -1, 2, 7, 8, -5, -6], [28, 27, -26, -25, 32, -31, 30, -29, 20, 19, -18, 17, 24, -23, 22, -21, -12, 11, -10, -9, 16, -15, 14, -13, 4, 3, -2, -1, 8, -7, 6, -5], [29, -30, -31, -32, -25, 26, 27, 28, 21, -22, -23, -24, 17, 18, 19, 20, -13, -14, -15, -16, -9, 10, 11, 12, 5, -6, -7, -8, -1, 2, 3, 4], [30, 29, -32, 31, -26, -25, -28, 27, 22, 21, -24, 23, -18, 17, -20, 19, -14, 13, -16, 15, -10, -9, -12, 11, 6, 5, -8, 7, -2, -1, -4, 3], [31, 32, 29, -30, -27, 28, -25, -26, 23, 24, 21, -22, -19, 20, 17, -18, -15, 16, 13, -14, -11, 12, -9, -10, 7, 8, 5, -6, -3, 4, -1, -2], [32, -31, 30, 29, -28, -27, 26, -25, 24, -23, 22, 21, -20, -19, 18, 17, -16, -15, 14, 13, -12, -11, 10, -9, 8, -7, 6, 5, -4, -3, 2, -1]]

"""@Dai Quoc Nguyen"""
'''Make a Hamilton matrix for quaternion linear transformations'''
def make_quaternion_mul(weight_list):
    """" The constructed 'hamilton' W is a modified version of the quaternion representation,
        thus doing tf.matmul(Input,W) is equivalent to W * Inputs. """
    # dim = kernel.size(1)//4
    # r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    # r2 = torch.cat([r, -i, -j, -k], dim=0)  # 0, 1, 2, 3
    # i2 = torch.cat([i, r, -k, j], dim=0)  # 1, 0, 3, 2
    # j2 = torch.cat([j, k, r, -i], dim=0)  # 2, 3, 0, 1
    # k2 = torch.cat([k, -j, i, r], dim=0)  # 3, 2, 1, 0
    # hamilton = torch.cat([r2, i2, j2, k2], dim=1)
    dim=weight_list.size(1)//32
    dim_list=[gg for gg in torch.split(weight_list,[dim]*32,dim=1)]
    cat_kernels_list = []
    for i in range(32):
        tempt = []
        for j in range(32):
            if kernel_list[j][i] > 0:
                tempt.append(dim_list[kernel_list[j][i] - 1])
            else:
                tempt.append(-dim_list[-kernel_list[j][i] - 1])
        tempt = torch.cat(tempt, dim=0)
        cat_kernels_list.append(tempt)
    cat_kernels_list = torch.cat(cat_kernels_list, dim=1)
    # assert kernel.size(1) == hamilton.size(1)
    return cat_kernels_list

'''Quaternion graph neural networks! QGNN layer for other downstream tasks!'''
class QGNNLayer_v2(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(QGNNLayer_v2, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        self.weight = Parameter(torch.FloatTensor(self.in_features//4, self.out_features))
        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        support = torch.mm(input, hamilton)  # Hamilton product, quaternion multiplication!
        output = torch.spmm(adj, support)
        output = self.bn(output)  # using act torch.tanh with BatchNorm can produce competitive results
        return self.act(output)

'''Quaternion graph neural networks! QGNN layer for node and graph classification tasks!'''
class QGNNLayer32(Module):
    def __init__(self, in_features, out_features, dropout, gate=None,quaternion_ff=True, act=F.relu):
        super(QGNNLayer32, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff 
        self.act =act
        self.dropout = nn.Dropout(dropout)
        self.bn = torch.nn.BatchNorm1d(out_features)
        #

        if self.quaternion_ff:
            self.weight = Parameter(torch.FloatTensor(self.in_features//32, self.out_features))
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

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)
class GATE(Module):
    def __init__(self, in_features, out_features, dropout, gate=None,quaternion_ff=True, act=F.relu):
        super(GATE, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.quaternion_ff = quaternion_ff
        self.act =act
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

            support = torch.mm(x, hamilton)  # Hamilton product, quaternion multiplication!
        else:
            support = torch.mm(x, self.weight)

        if double_type_used_in_graph: #to deal with scalar type between node and graph classification tasks, caused by pre-defined feature inputs
            support = support.double()

        output = torch.spmm(adj, support)

        # output = self.bn(output) # should tune whether using BatchNorm or Dropout

        return self.act(output)
'''Dual quaternion multiplication'''
def dual_quaternion_mul(A, B, input):
    '''(A, B) * (C, D) = (A * C, A * D + B * C)'''
    dim = input.size(1) // 2
    C, D = torch.split(input, [dim, dim], dim=1)
    A_hamilton = make_quaternion_mul(A)
    B_hamilton = make_quaternion_mul(B)
    AC = torch.mm(C, A_hamilton)
    AD = torch.mm(D, A_hamilton)
    BC = torch.mm(C, B_hamilton)
    AD_plus_BC = AD + BC
    return torch.cat([AC, AD_plus_BC], dim=1)

''' Dual Quaternion Graph Neural Networks! https://arxiv.org/abs/2104.07396 '''
class DQGNN_layer(Module):
    def __init__(self, in_features, out_features, act=torch.tanh):
        super(DQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.act = act
        #
        self.A = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2)) # (A, B) = A + eB, e^2 = 0
        self.B = Parameter(torch.FloatTensor(self.in_features // 8, self.out_features // 2))

        self.reset_parameters()
        self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.A.size(0) + self.A.size(1)))
        self.A.data.uniform_(-stdv, stdv)
        self.B.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = dual_quaternion_mul(self.A, self.B, input)
        output = torch.spmm(adj, support)
        output = self.bn(output)
        return self.act(output)

''' Simplifying Quaternion Graph Neural Networks! following SGC https://arxiv.org/abs/1902.07153'''
class SQGNN_layer(Module):
    def __init__(self, in_features, out_features, step_k=1):
        super(SQGNN_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.step_k = step_k
        self.weight = Parameter(torch.FloatTensor(self.in_features // 4, self.out_features))
        self.reset_parameters()
        #self.bn = torch.nn.BatchNorm1d(out_features)

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        hamilton = make_quaternion_mul(self.weight)
        new_input = torch.spmm(adj, input)
        if self.step_k > 1:
            for _ in range(self.step_k-1):
                new_input = torch.spmm(adj, new_input)
        output = torch.mm(new_input, hamilton)  # Hamilton product, quaternion multiplication!
        #output = self.bn(output)
        return output
