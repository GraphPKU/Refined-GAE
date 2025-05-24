import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
import argparse
import itertools
import wandb
import time
import math

from torch.utils.data import DataLoader
from ogb.linkproppred import PygLinkPropPredDataset, Evaluator
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import negative_sampling, add_self_loops, to_undirected, train_test_split_edges
from torch_geometric.data import Data
from torch_sparse import SparseTensor
from util import adjoverlap
from torch_sparse.matmul import spmm_add
import numpy as np
import scipy.sparse as sp

from torch_geometric.datasets import Planetoid

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-collab', 
                        choices=['ogbl-ddi','ogbl-collab', 'ogbl-ppa', 'ogbl-citation2', 'Cora', 'CiteSeer', 'PubMed'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--emb_hidden", default=64, type=int)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--interval", default=100, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=True)
    parser.add_argument("--metric", default='hits@20', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--relu", action='store_true', default=False)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", default='GCN', 
                        choices=['GCN', 'LightGCN'], type=str)
    parser.add_argument("--maskinput", action='store_true', default=False)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float)
    parser.add_argument("--dpe", default=0, type=float)
    parser.add_argument("--drop_edge", action='store_true', default=False)
    parser.add_argument("--residual", default=0, type=float)
    parser.add_argument("--mlp_layers", default=2, type=int)
    parser.add_argument("--pred", default='mlp', 
                        choices=['dot', 'mlp', 'ncn'], type=str)
    parser.add_argument("--res", action='store_true', default=False)
    parser.add_argument("--conv", default='GCN', type=str)
    parser.add_argument("--use_valid_as_input", action='store_true', default=False)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--exp', action='store_true', default=False)
    parser.add_argument('--scale', default=False, action='store_true')
    parser.add_argument('--beta', default=1.0, type=float)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--optimizer', default='adam', type=str)

    args = parser.parse_args()
    return args

class DotPredictor(nn.Module):
    def __init__(self):
        super(DotPredictor, self).__init__()

    def forward(self, x_i, x_j):
        # x_i, x_j shape: [batch_size, hidden_dim]
        return (x_i * x_j).sum(dim=-1)  # dot product

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False, scale=False, norm=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        self.ln = nn.LayerNorm(h_feats)
        self.norm = norm

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.scale:
            x = x / (torch.norm(x, dim=-1, keepdim=True) + 1e-5)
        ori = x
        for lin in self.lins[:-1]:
            x = lin(x)
            if self.res:
                x += ori
            if self.norm:
                x = self.ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()
    
class GCN(nn.Module):
        
    def __init__(self, in_feats, h_feats, relu=False, prop_step=2, dropout=0.2, residual=0):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
    
    def forward(self, adj, in_feat):
        ori = in_feat
        h = self.conv1(in_feat, adj).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self.conv2(h, adj).flatten(1) + self.residual * ori
        return h
    
class LightGCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, alpha = 0.5, exp = False, relu = False, norm=False):
        super(LightGCN, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats, weight=True, bias=False)
        self.conv2 = GCNConv(h_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.relu = relu
        self.alpha = alpha
        if exp:
            self.alphas = nn.Parameter(alpha ** torch.arange(prop_step))
        else:
            self.alphas = nn.Parameter(torch.ones(prop_step))
        self.norm = norm
        if self.norm:
            self.ln = nn.LayerNorm(h_feats)
            self.dp = nn.Dropout(dropout)

    def _apply_norm_and_activation(self, x):
        if self.norm:
            x = self.ln(x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, adj, in_feat):
        h = self.conv1(in_feat, adj).flatten(1)
        res = h * self.alphas[0]
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h)
            h = self.conv2(h, adj).flatten(1)
            res += h * self.alphas[i]
        return res

class GCN_with_feature(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, residual = 0, relu = False, norm=False, conv='GCN'):
        super(GCN_with_feature, self).__init__()
        self.conv1 = GCNConv(in_feats, h_feats)
        self.conv2 = GCNConv(h_feats, h_feats)
        self.prop_step = prop_step
        self.residual = residual
        self.dropout = dropout
        self.relu = relu
        self.norm = norm
        if self.norm:
            self.ln = nn.LayerNorm(h_feats)
            self.dp = nn.Dropout(dropout)
    
    def _apply_norm_and_activation(self, x):
        if self.norm:
            x = self.ln(x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, adj, in_feat):
        x = self.conv1(in_feat, adj)
        ori = x
        for i in range(self.prop_step - 1):
            x = self._apply_norm_and_activation(x)
            x = self.conv2(x, adj) + self.residual * ori
        return x

class NCNPredictor(nn.Module):
    def __init__(self,
                 in_channels,
                 hidden_channels,
                 out_channels,
                 dropout,
                 beta = 1.0,
                 mlp_layers = 3,
                 res = False, 
                 norm = False):
        super().__init__()

        #self.register_parameter("beta", nn.Parameter(beta*torch.ones((1))))
        self.beta = beta
        self.norm = norm
        if self.norm:
            self.xcnlin = nn.Sequential(
                nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
                nn.ReLU(), nn.Dropout(dropout), 
                nn.Linear(hidden_channels, hidden_channels))
            self.xijlin = nn.Sequential(
                nn.Linear(in_channels, hidden_channels), nn.LayerNorm(hidden_channels),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels))
        else:
            self.xcnlin = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Dropout(dropout), 
                nn.Linear(hidden_channels, hidden_channels))
            self.xijlin = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden_channels, hidden_channels))
        self.lin = nn.ModuleList()
        for _ in range(mlp_layers - 1):
            self.lin.append(nn.Linear(hidden_channels, hidden_channels))
        self.lin.append(nn.Linear(hidden_channels, out_channels))
        self.ln = nn.LayerNorm(hidden_channels) if norm else nn.Identity()
        self.dropout = dropout
        self.res = res

    def forward(self, x, adj_t, tar_ei):
        xi = x[tar_ei[0]]
        xj = x[tar_ei[1]]
        cn = adjoverlap(adj_t, adj_t, tar_ei)
        xcn = spmm_add(cn, x)
        xij = self.xijlin(xi * xj)
        xcn = self.xcnlin(xcn)

        x = self.beta * xcn + xij

        ori = x

        for lin in self.lin[:-1]:
            x = lin(x)
            if self.res:
                x = x + ori
            if self.norm:
                x = self.ln(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lin[-1](x)
 
        return x.squeeze()

def eval_hits(y_pred_pos, y_pred_neg, K):
    '''
        compute Hits@K
        For each positive target node, the negative target nodes are the same.

        y_pred_neg is an array.
        rank y_pred_pos[i] against y_pred_neg for each i
    '''

    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}

    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)

    return {'hits@{}'.format(K): hitsK}

def eval_mrr(y_pred_pos, y_pred_neg):
    '''
        compute mrr
        y_pred_neg is an array with shape (batch size, num_entities_neg).
        y_pred_pos is an array with shape (batch size, )
    '''


    # calculate ranks
    y_pred_pos = y_pred_pos.view(-1, 1)
    # optimistic rank: "how many negatives have a larger score than the positive?"
    # ~> the positive is ranked first among those with equal score
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    # pessimistic rank: "how many negatives have at least the positive score?"
    # ~> the positive is ranked last among those with equal score
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1./ranking_list.to(torch.float)

    return {'mrr_list': mrr_list}

def adjustlr(optimizer, decay_ratio, init_lr):
    lr_ = init_lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, predictor, data, train_pos_edge, optimizer, args, embedding=None):
    """
    train_pos_edge: shape [num_train_edges, 2]
    """
    model.train()
    predictor.train()
    device = train_pos_edge.device
    # We'll do a simple mini-batch approach
    total_loss = 0
    loader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    if args.maskinput:
        adjmask = torch.ones(train_pos_edge.size(0), dtype=torch.bool, device=device)

    for perm in loader:
        # mask out edges if needed (args.maskinput), else just use original graph
        if embedding is not None:
            if data.x is not None:
                xemb = torch.cat([data.x, embedding.weight], dim=1)
            else:
                xemb = embedding.weight
        else:
            xemb = data.x
        if args.maskinput:
            adjmask[perm] = False
            edge = train_pos_edge[adjmask].t()
            # edge = add_self_loops(edge, num_nodes=data.num_nodes)[0]
            adj = SparseTensor.from_edge_index(edge,sparse_sizes=(data.num_nodes, data.num_nodes)).to_device(train_pos_edge.device)
            adj = adj.to_symmetric().coalesce()
            adjmask[perm] = True
        else:
            adj = data.adj
        
        # model forward
        h = model(adj, xemb)

        pos_edge = train_pos_edge[perm]
        # negative sampling
        neg_edge_now = negative_sampling(data.edge_index, num_nodes=data.num_nodes, num_neg_samples=args.num_neg * pos_edge.size(0)).t().to(device)

        if args.pred == 'ncn':
            pos_score = predictor(h, adj, pos_edge.t())
            neg_score = predictor(h, adj, neg_edge_now.t())
        else:
            pos_score = predictor(h[pos_edge[:,0]], h[pos_edge[:,1]])
            neg_score = predictor(h[neg_edge_now[:,0]], h[neg_edge_now[:,1]])

        loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))

        optimizer.zero_grad()
        loss.backward()
        if args.dataset != 'ogbl-ppa':
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            nn.utils.clip_grad_norm_(predictor.parameters(), args.clip_norm)
            if embedding is not None:
                nn.utils.clip_grad_norm_(embedding.parameters(), args.clip_norm)

        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)

@torch.no_grad()
def test(model, predictor, data, pos_test_edge, neg_test_edge, args, embedding=None):
    model.eval()
    predictor.eval()

    if embedding is not None:
        if data.x is not None:
            xemb = torch.cat([data.x, embedding.weight], dim=1)
        else:
            xemb = embedding.weight
    else:
        xemb = data.x

    h = model(data.full_adj, xemb)

    # predict
    # break into mini-batches for large edge sets
    pos_dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size)
    neg_dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size)

    pos_score = []
    for idx in pos_dataloader:
        e = pos_test_edge[idx]
        out = predictor(h[e[:,0]], h[e[:,1]]) if args.pred != 'ncn' else predictor(h, data.full_adj, e.t())
        pos_score.append(out)
    pos_score = torch.cat(pos_score, dim=0)

    neg_score = []
    for idx in neg_dataloader:
        e = neg_test_edge[idx]
        out = predictor(h[e[:,0]], h[e[:,1]]) if args.pred != 'ncn' else predictor(h, data.full_adj, e.t())
        neg_score.append(out)
    neg_score = torch.cat(neg_score, dim=0)

    results = {}
    if args.metric == 'mrr':
        neg_score = neg_score.view(-1, 1000)
        results['mrr'] = eval_mrr(pos_score, neg_score)['mrr_list'].mean().item()
    else:
        for k in [20, 50, 100]:
            results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']

    return results

@torch.no_grad()
def evaluate(model, predictor, data, pos_train_edge, pos_valid_edge, neg_valid_edge, args, embedding=None):
    """
    Evaluate on validation set + some measure on training edges
    """
    model.eval()
    predictor.eval()

    if embedding is not None:
        if data.x is not None:
            xemb = torch.cat([data.x, embedding.weight], dim=1)
        else:
            xemb = embedding.weight
    else:
        xemb = data.x

    h = model(data.adj, xemb)

    # Validation
    valid_pos_score = []
    valid_dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
    for idx in valid_dataloader:
        e = pos_valid_edge[idx]
        out = predictor(h[e[:,0]], h[e[:,1]]) if args.pred != 'ncn' else predictor(h, data.adj, e.t())
        valid_pos_score.append(out)
    valid_pos_score = torch.cat(valid_pos_score, dim=0)

    valid_neg_score = []
    neg_dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size)
    for idx in neg_dataloader:
        e = neg_valid_edge[idx]
        out = predictor(h[e[:,0]], h[e[:,1]]) if args.pred != 'ncn' else predictor(h, data.adj, e.t())
        valid_neg_score.append(out)
    valid_neg_score = torch.cat(valid_neg_score, dim=0)

    valid_results = {}
    if args.metric == 'mrr':
        valid_neg_score = valid_neg_score.view(-1, 1000)
        valid_results['mrr'] = eval_mrr(valid_pos_score, valid_neg_score)['mrr_list'].mean().item()
    else:
        for k in [20, 50, 100]:
            valid_results[f'hits@{k}'] = eval_hits(valid_pos_score, valid_neg_score, k)[f'hits@{k}']


    # "Train" hits for reference
    train_pos_score = []
    train_dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
    # This is partly for debugging: compare how well the model does on training edges
    for idx in train_dataloader:
        e = pos_train_edge[idx]
        out = predictor(h[e[:,0]], h[e[:,1]]) if args.pred != 'ncn' else predictor(h, data.adj, e.t())
        train_pos_score.append(out)
    train_pos_score = torch.cat(train_pos_score, dim=0)

    # For a fair comparison, we reuse valid_neg_score. 
    # But logically, you'd want negative samples for the training edges specifically.
    train_results = {}
    if args.metric == 'mrr':
        train_neg_score = valid_neg_score.view(-1, 1000)
        train_results['mrr'] = eval_mrr(train_pos_score, train_neg_score)['mrr_list'].mean().item()
    else:
        for k in [20, 50, 100]:
            train_results[f'hits@{k}'] = eval_hits(train_pos_score, valid_neg_score, k)[f'hits@{k}']

    return valid_results, train_results

def random_split_edges(data, val_ratio=0.1, test_ratio=0.2):
    result = train_test_split_edges(data, val_ratio=val_ratio, test_ratio=test_ratio)

    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = result.train_pos_edge_index.t()
    split_edge['valid']['edge'] = result.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = result.val_neg_edge_index.t()
    split_edge['test']['edge'] = result.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = result.test_neg_edge_index.t()
    return split_edge

def load_data(dataset):
    dataset = Planetoid(root='dataset', name=dataset)
    data = dataset[0]
    split_edge = random_split_edges(data)
    data.edge_index = to_undirected(split_edge['train']['edge'].t())
    data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
    data.num_nodes = data.x.shape[0]
    data.edge_weight = None
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    return data, split_edge

def load_data1(dataset):
    
    # Load dataset (Planetoid example)
    dataset = Planetoid(root='dataset', name=dataset)
    data = dataset[0]
    
    # Assume data.edge_index is of shape [2, num_edges]
    num_edges = data.edge_index.size(1)
    num_nodes = data.num_nodes
    
    # --- 1. Split positive edges ---
    # Create a permutation of edge indices
    eids = np.arange(num_edges)
    np.random.shuffle(eids)
    
    valid_size = int(num_edges * 0.1)
    test_size  = int(num_edges * 0.2)
    train_size = num_edges - valid_size - test_size

    # Use the permuted indices to split the edge_index
    test_pos_edge_index  = data.edge_index[:, eids[:test_size]]
    valid_pos_edge_index = data.edge_index[:, eids[test_size:test_size + valid_size]]
    train_pos_edge_index = data.edge_index[:, eids[test_size + valid_size:]]
    
    # --- 2. Generate negative edges ---
    # Build a dense (binary) adjacency matrix from the positive edges.
    row = data.edge_index[0].numpy()
    col = data.edge_index[1].numpy()
    # Create a sparse matrix with ones at the locations of positive edges.
    adj = sp.coo_matrix((np.ones(num_edges), (row, col)), shape=(num_nodes, num_nodes))
    # Convert to a dense matrix, subtract self-loops (via identity) and positive edges.
    adj_dense = adj.todense()
    adj_neg = 1 - adj_dense - np.eye(num_nodes)
    
    # Find indices of non-existent edges (i.e. candidate negatives)
    neg_u, neg_v = np.where(adj_neg != 0)
    
    # Sample as many negative edges as there are positive edges.
    neg_eids = np.random.choice(len(neg_u), num_edges, replace=True)
    test_neg_edge_index  = torch.tensor(
        [neg_u[neg_eids[:test_size]], neg_v[neg_eids[:test_size]]],
        dtype=torch.long
    )
    valid_neg_edge_index = torch.tensor(
        [neg_u[neg_eids[test_size:test_size + valid_size]], neg_v[neg_eids[test_size:test_size + valid_size]]],
        dtype=torch.long
    )
    train_neg_edge_index = torch.tensor(
        [neg_u[neg_eids[test_size + valid_size:]], neg_v[neg_eids[test_size + valid_size:]]],
        dtype=torch.long
    )
    
    # --- 3. Build the "message passing" graphs ---
    # The DGL code builds two graphs:
    #   - One using only the training positive edges.
    #   - One using both training and validation positive edges.
    # In PyG we mimic this by constructing two edge_index tensors and adding self-loops
    # and converting to an undirected graph.
    
    # (a) Graph for training only
    graph_train_edge_index = to_undirected(train_pos_edge_index)
    graph_train_edge_index, _ = add_self_loops(graph_train_edge_index, num_nodes=num_nodes)
    
    # (b) Graph for training + validation (often used during message passing at test time)
    train_valid_edge_index = torch.cat((train_pos_edge_index, valid_pos_edge_index), dim=1)
    graph_full_edge_index = to_undirected(train_valid_edge_index)
    graph_full_edge_index, _ = add_self_loops(graph_full_edge_index, num_nodes=num_nodes)
    
    # Optionally, attach these graphs to the data object for later use.
    # (Here we use new attributes; your usage may vary.)
    data.train_edge_index = graph_train_edge_index
    data.train_valid_edge_index = graph_full_edge_index

    # (c) Rebuild a symmetric sparse adjacency for the training graph (if needed)
    data.adj_t = SparseTensor.from_edge_index(graph_train_edge_index,
                                                sparse_sizes=(num_nodes, num_nodes))
    data.adj_t = data.adj_t.to_symmetric().coalesce()
    
    # --- 4. Pack the edge splits into a dictionary ---
    split_edge = {
        'train': {
            'edge': train_pos_edge_index.t(),    # positive training edges
            'edge_neg': train_neg_edge_index.t() # corresponding negative samples
        },
        'valid': {
            'edge': valid_pos_edge_index.t(),    # positive validation edges
            'edge_neg': valid_neg_edge_index.t()
        },
        'test': {
            'edge': test_pos_edge_index.t(),     # positive test edges
            'edge_neg': test_neg_edge_index.t()
        }
    }
    
    return data, split_edge

def load_data2(dataset):

    dataset = Planetoid(root='dataset', name=dataset)
    data = dataset[0]
    num_nodes = data.num_nodes
    row, col = data.edge_index
    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]
    n_v = int(math.floor(0.1 * row.size(0)))
    n_t = int(math.floor(0.2 * row.size(0)))
    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]
    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)
    # Negative edges (cannot guarantee (i,j) and (j,i) won't both appear)
    neg_edge_index = negative_sampling(
        data.edge_index, num_nodes=num_nodes,
        num_neg_samples=row.size(0))
    data.val_neg_edge_index = neg_edge_index[:, :n_v]
    data.test_neg_edge_index = neg_edge_index[:, n_v:n_v + n_t]
    data.train_neg_edge_index = neg_edge_index[:, n_v + n_t:]
    split_edge = {'train': {}, 'valid': {}, 'test': {}}
    split_edge['train']['edge'] = data.train_pos_edge_index.t()
    split_edge['train']['edge_neg'] = data.train_neg_edge_index.t()
    split_edge['valid']['edge'] = data.val_pos_edge_index.t()
    split_edge['valid']['edge_neg'] = data.val_neg_edge_index.t()
    split_edge['test']['edge'] = data.test_pos_edge_index.t()
    split_edge['test']['edge_neg'] = data.test_neg_edge_index.t()
    data.edge_index = data.train_pos_edge_index
    data.edge_index = add_self_loops(data.edge_index, num_nodes=num_nodes)[0]
    data.edge_index = to_undirected(data.edge_index)
    data.adj_t = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(num_nodes, num_nodes))

    return data, split_edge

def main():
    args = parse()
    print(args)
    wandb.init(project='Refined-GAE', config=args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    if args.dataset.startswith('ogbl-'):
        # 1) Load dataset from OGB
        dataset = PygLinkPropPredDataset(name=args.dataset)
        split_edge = dataset.get_edge_split()
        data = dataset[0]
    else:
        data, split_edge = load_data2(args.dataset)
        print(data.adj_t, flush=True)
        print(split_edge['train']['edge'].shape, flush=True)
        print(split_edge['valid']['edge'].shape, flush=True)
        print(split_edge['test']['edge'].shape, flush=True)

    # 2) We only keep training edges from year >= 2011 in 'ogbl-collab'
    if args.dataset == 'ogbl-collab':
        selected_year_index = torch.reshape(
            (split_edge['train']['year'] >= 2011).nonzero(as_tuple=False), (-1,))
        split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
        split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
        split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]

        edge = to_undirected(split_edge['train']['edge'].t())
        data.edge_index = add_self_loops(edge, num_nodes=data.num_nodes)[0]
        data.adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))

        full_edge_index = torch.cat([split_edge['valid']['edge'].t(), split_edge['train']['edge'].t()], dim=-1)
        full_edge_index = to_undirected(full_edge_index)
        full_edge_index = add_self_loops(full_edge_index, num_nodes=data.num_nodes)[0]
        data.full_adj = SparseTensor.from_edge_index(full_edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj = data.adj.to(device)
        data.full_adj = data.full_adj.to(device)
    elif args.dataset == 'ogbl-citation2':
        for name in ['train','valid','test']:
            u=split_edge[name]["source_node"]
            v=split_edge[name]["target_node"]
            split_edge[name]['edge']=torch.stack((u,v),dim=0).t()
        for name in ['valid','test']:
            u=split_edge[name]["source_node"].repeat(1, 1000).view(-1)
            v=split_edge[name]["target_node_neg"].view(-1)
            split_edge[name]['edge_neg']=torch.stack((u,v),dim=0).t()  
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.adj = data.adj.to_symmetric().coalesce()
        data.adj = data.adj.to(device)
        data.full_adj = data.adj
    elif args.dataset == 'ogbl-ppa' or args.dataset == 'ogbl-ddi':
        data.edge_index = add_self_loops(data.edge_index, num_nodes=data.num_nodes)[0]
        data.adj = SparseTensor.from_edge_index(data.edge_index, sparse_sizes=(data.num_nodes, data.num_nodes))
        data.x = None
        data.adj = data.adj.to(device)
        data.full_adj = data.adj
    else:
        data.adj = data.adj_t.to(device)
        data.full_adj = data.adj

    # Move the feature matrix to GPU if available
    data.x = data.x.to(device) if data.x is not None else None
    data.edge_index = data.edge_index

    # By default, PyG's Data object doesn't keep adjacency shape in `data.num_nodes` 
    # but we should store it or compute from dataset
    data.num_nodes = data.x.size(0) if data.x is not None else dataset[0].num_nodes

    # Convert edges to device
    train_pos_edge = split_edge['train']['edge'].to(device)
    valid_pos_edge = split_edge['valid']['edge'].to(device)
    valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
    test_pos_edge = split_edge['test']['edge'].to(device)
    test_neg_edge = split_edge['test']['edge_neg'].to(device)

    # 3) Possibly use an embedding if args.emb_hidden > 0
    embedding = None
    if args.emb_hidden > 0:
        embedding = nn.Embedding(data.num_nodes, args.emb_hidden)
        nn.init.orthogonal_(embedding.weight)
        embedding = embedding.to(device)

    # 4) Build predictor
    if args.pred == 'dot':
        predictor = DotPredictor().to(device)
    elif args.pred == 'mlp':
        predictor = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.res, args.scale, args.norm).to(device)
    elif args.pred == 'ncn':
        predictor = NCNPredictor(args.hidden, args.hidden, 1, args.dropout, args.beta, args.mlp_layers, args.res, args.norm).to(device)
    else:
        raise NotImplementedError

    # 5) Build model
    in_feats = data.x.size(-1) + args.emb_hidden if data.x is not None else args.emb_hidden

    if args.model == 'GCN':
        if args.dataset == 'ogbl-ddi' or args.dataset == 'ogbl-ppa':
            model = GCN(
                in_feats = in_feats,
                h_feats = args.hidden,
                prop_step = args.prop_step,
                dropout = args.dropout,
                relu = args.relu,
                residual = args.residual
            ).to(device)
        else:
            model = GCN_with_feature(
                in_feats = in_feats,
                h_feats = args.hidden,
                prop_step = args.prop_step,
                dropout = args.dropout,
                residual = args.residual,
                relu = args.relu,
                norm = args.norm,
                conv = args.conv
            ).to(device)
    elif args.model == 'LightGCN':
        model = LightGCN(
            in_feats = in_feats,
            h_feats = args.hidden,
            prop_step = args.prop_step,
            dropout = args.dropout,
            alpha = args.alpha,
            exp = args.exp,
            relu = args.relu,
            norm = args.norm
        ).to(device)
    else:
        raise NotImplementedError

    # 6) Setup optimizer
    parameters = itertools.chain(model.parameters(), predictor.parameters())
    if embedding is not None:
        parameters = itertools.chain(parameters, embedding.parameters())
    optimizer = torch.optim.Adam(parameters, lr=args.lr)

    best_val = 0
    final_test_result = None
    best_epoch = 0

    for epoch in range(args.epochs):
        # train
        loss = train(model, predictor, data, train_pos_edge, optimizer, args, embedding)
        
        # adjust lr
        if epoch % args.interval == 0 and args.step_lr_decay:
            adjustlr(optimizer, epoch / args.epochs, args.lr)
        
        # evaluate
        valid_results, train_results = evaluate(
            model, predictor, data, train_pos_edge, valid_pos_edge, valid_neg_edge, args, embedding
        )
        
        # for printing
        for k, v in valid_results.items():
            print(f'Epoch {epoch}, Validation {k}: {v:.4f}')
        for k, v in train_results.items():
            print(f'Epoch {epoch}, Train {k}: {v:.4f}')

        test_results = test(model, predictor, data, test_pos_edge, test_neg_edge, args, embedding)

        for k, v in test_results.items():
            print(f'Epoch {epoch}, Test {k}: {v:.4f}')

        if args.dataset == 'ogbl-ddi' and train_results[args.metric] >= 0.90:
            break

        # track best
        if valid_results[args.metric] > best_val:
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results

        if epoch - best_epoch >= 200:
            # early stop
            break

        print(f"Epoch {epoch}, Loss: {loss:.4f}, "
              f"Train {args.metric}: {train_results[args.metric]:.4f}, "
              f"Valid {args.metric}: {valid_results[args.metric]:.4f}, "
              f"Test {args.metric}: {test_results[args.metric]:.4f}")

        wandb.log({
            'epoch': epoch,
            'loss': loss, 
            'train_hit': train_results[args.metric],
            'valid_hit': valid_results[args.metric],
            'test_hit': test_results[args.metric]
        })

    print(f"Best Test hit@20: {final_test_result[args.metric]:.4f}")
    wandb.log({'final_test_hit': final_test_result[args.metric]})

if __name__ == "__main__":
    main()
