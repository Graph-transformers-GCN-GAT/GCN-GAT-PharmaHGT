import dgl
import torch
from dgl.nn.pytorch import GraphConv, GATConv

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F

def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)

###############################################################################

class GCNReg(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, saliency=False):
        super(GCNReg, self).__init__()
        self.conv1 = GraphConv(in_dim, hidden_dim)
        self.conv2 = GraphConv(hidden_dim, hidden_dim)

        self.classify1 = nn.Linear(hidden_dim, hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g):
        # Use node degree as the initial node feature. For undirected graphs, the in-degree
        # is the same as the out_degree.

        if torch.cuda.is_available():
            h = g.ndata['h'].float().cuda()
        else:
            h = g.ndata['h'].float()

        if self.saliency == True:
            h.requires_grad = True
        h1 = F.relu(self.conv1(g, h))
        h1 = F.relu(self.conv2(g, h1))

        g.ndata['h'] = h1
        # Calculate graph representation by averaging all the node representations.
        hg = dgl.mean_nodes(g, 'h')
        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = self.classify3(output)
        if self.saliency == True:
            output.backward()
            return output, h.grad
        else:
            return output



def collate(samples):
    graphs, descriptors, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(descriptors), torch.tensor(labels)

###############################################################################

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes, feature_dim=None, num_heads=8, saliency=False): #change number of head from 4 to 8
        super(GAT, self).__init__()
        self.num_heads = num_heads
        self.conv1 = GATConv(in_dim, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.conv2 = GATConv(hidden_dim * num_heads, hidden_dim, num_heads=num_heads, allow_zero_in_degree=True)
        self.dropout = nn.Dropout(0.5)  # Add dropout for regularization
        self.classify1 = nn.Linear(hidden_dim * num_heads + (feature_dim if feature_dim else 0), hidden_dim)
        self.classify2 = nn.Linear(hidden_dim, hidden_dim)
        self.classify3 = nn.Linear(hidden_dim, n_classes)
        self.saliency = saliency

    def forward(self, g, features=None):
        
        if torch.cuda.is_available():
            g = g.to('cuda')

        h = g.ndata['h'].float()
        if torch.cuda.is_available():
            h = h.cuda()

        if self.saliency and not h.requires_grad:
            h.requires_grad = True

        h1 = F.relu(self.conv1(g, h).flatten(1))
        h1 = F.relu(self.conv2(g, h1).flatten(1))

        if self.saliency:
            h1.retain_grad()

        g.ndata['h'] = h1
        hg = dgl.mean_nodes(g, 'h')

        if features is not None:
            hg = torch.cat([hg, features], dim=1)

        hg = self.dropout(hg)  # Apply dropout before classification layers

        output = F.relu(self.classify1(hg))
        output = F.relu(self.classify2(output))
        output = self.classify3(output)

        if self.saliency:
            return output, h1
        else:
            return output




