#\cite{poli2019graph}一个亮点是保留了Cora图的有向性，其他的工作往往对邻接矩阵$A$做变换$A\leftarrow A+A^T$，强制把有方向的Cora图变成了无向对称图，但是\cite{poli2019graph}没有这样做；此时，$A$的列代表边的出发节点、行代表边的目的地节点，拉普拉斯矩阵$L$和标准化拉普拉斯矩阵$\mathcal{L}$的定义也针对有向图做了调整，可以看出来作者使用的是入度矩阵$D_\text{in}$

import dgl
import dgl.function as fn
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy.sparse as spp
import matplotlib.pyplot as plt
import networkx as nx

edgelist=[(0,4),(0,1),(4,1),(4,3),(1,2),(3,2),(3,5)]
g=nx.DiGraph(edgelist)

# add self-edge for each node
g.remove_edges_from(nx.selfloop_edges(g))
g.add_edges_from(zip(g.nodes(), g.nodes()))

#nx.draw(g, with_labels=True)
#plt.show()

g = dgl.DGLGraph(g)

degs = g.in_degrees().float()
norm = torch.pow(degs, -0.5)
norm[torch.isinf(norm)] = 0
g.ndata['norm'] = norm.unsqueeze(1)

h=torch.ones([6, 2])
# normalization by square root of src degree
h = h * g.ndata['norm']
g.ndata['h'] = h
g.update_all(fn.copy_src(src='h', out='m'), fn.sum(msg='m', out='h'))
h = g.ndata.pop('h')
# normalization by square root of dst degree
h = h * g.ndata['norm']

print(h)