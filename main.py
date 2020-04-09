import math
import numpy as np
import scipy.sparse as sp
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

from torchgde import GCN, GCNLayer, ODEBlock, GDEFunc, PerformanceContainer, accuracy

import dgl
import dgl.data
import networkx as nx

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# seed for repeatability
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.manual_seed(0)
np.random.seed(0)

data = dgl.data.CoraDataset()

X = torch.FloatTensor(data.features).to(device)
Y = torch.LongTensor(data.labels).to(device)

train_mask = torch.BoolTensor(data.train_mask)
val_mask = torch.BoolTensor(data.val_mask)
test_mask = torch.BoolTensor(data.test_mask)

num_feats = X.shape[1]
n_classes = data.num_labels

g = data.graph
g.remove_edges_from(nx.selfloop_edges(g))
g.add_edges_from(zip(g.nodes(), g.nodes()))

g = dgl.DGLGraph(g)
edges = g.edges()
n_edges = g.number_of_edges()

degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0

g.ndata['norm'] = norm.unsqueeze(1).to(device)

gnn = nn.Sequential(
    GCNLayer(g=g, in_feats=64, out_feats=64, activation=nn.Softplus(), dropout=0.9),
    GCNLayer(g=g, in_feats=64, out_feats=64, activation=None, dropout=0.9)
).to(device)

gdefunc = GDEFunc(gnn)

gde = ODEBlock(odefunc=gdefunc, method='rk4', atol=1e-3, rtol=1e-4, adjoint=False).to(device)

m = nn.Sequential(
    GCNLayer(g=g, in_feats=num_feats, out_feats=64, activation=F.relu, dropout=0.4),
    gde,
    GCNLayer(g=g, in_feats=64, out_feats=n_classes, activation=None, dropout=0.)
).to(device)

opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=5e-4)
criterion = torch.nn.CrossEntropyLoss()
logger = PerformanceContainer(data={'train_loss':[], 'train_accuracy':[],
                                   'test_loss':[], 'test_accuracy':[],
                                   'forward_time':[], 'backward_time':[],
                                   'nfe': []})
steps = 3000
verbose_step = 150
num_grad_steps = 0



for i in range(steps): # looping over epochs
    m.train()
    start_time = time.time()

    outputs = m(X)
    f_time = time.time() - start_time

    nfe = m._modules['1'].odefunc.nfe

    y_pred = outputs

    loss = criterion(y_pred[train_mask], Y[train_mask])
    opt.zero_grad()
    
    start_time = time.time()
    loss.backward()
    b_time = time.time() - start_time
    
    opt.step()
    num_grad_steps += 1

    with torch.no_grad():
        m.eval()

        # calculating outputs again with zeroed dropout
        y_pred = m(X)
        m._modules['1'].odefunc.nfe = 0

        train_loss = loss.item()
        train_acc = accuracy(y_pred[train_mask], Y[train_mask]).item()
        test_acc = accuracy(y_pred[test_mask], Y[test_mask]).item()
        test_loss = criterion(y_pred[test_mask], Y[test_mask]).item()
        logger.deep_update(logger.data, dict(train_loss=[train_loss], train_accuracy=[train_acc],
                           test_loss=[test_loss], test_accuracy=[test_acc], nfe=[nfe], forward_time=[f_time],
                           backward_time=[b_time]))

    if num_grad_steps % verbose_step == 0:
        print('[{}], Loss: {:3.3f}, Train Accuracy: {:3.3f}, Test Accuracy: {:3.3f}, NFE: {}'.format(num_grad_steps,
                                                                                                    train_loss,
                                                                                                    train_acc,
                                                                                                    test_acc,
                                                                                                    nfe))
plt.figure(figsize=(16,8))
plt.subplot(2,2,1)
plt.plot(logger.data['train_loss'])
plt.plot(logger.data['test_loss'])
plt.legend(['Train loss', 'Test loss'])
plt.subplot(2,2,2)
plt.plot(logger.data['train_accuracy'])
plt.plot(logger.data['test_accuracy'])
plt.legend(['Train accuracy', 'Test accuracy'])
plt.subplot(2,2,3)
plt.plot(logger.data['forward_time'])
plt.plot(logger.data['backward_time'])
plt.legend(['Forward time', 'Backward time'])
plt.subplot(2,2,4)
plt.plot(logger.data['nfe'], marker='o', linewidth=0.1, markersize=1)
plt.legend(['NFE']);

