import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn import GraphConv, SAGEConv, GATConv, GINConv

class Hadamard_MLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False, norm=False, scale=False, act='relu'):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.scale = scale
        if scale:
            self.scale_norm = nn.LayerNorm(h_feats)
        self.norm = norm
        if norm:
            self.norms = torch.nn.ModuleList()
            for _ in range(layer - 1):
                self.norms.append(nn.LayerNorm(h_feats))
        if act == 'relu':
            self.act = F.relu
        elif act == 'gelu':
            self.act = F.gelu
        elif act == 'silu':
            self.act = F.silu
        else:
            raise ValueError('Activation function not supported')

    def forward(self, x_i, x_j):
        x = x_i * x_j
        if self.scale:
            x = self.scale_norm(x)
        ori = x
        for i in range(len(self.lins) - 1):
            x = self.lins[i](x)
            if self.res:
                x += ori
            if self.norm:
                x = self.norms[i](x)
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x.squeeze()
    
'''class AttMLPPredictor(nn.Module):
    def __init__(self, h_feats, dropout, layer=2, res=False):
        super().__init__()
        self.lins = torch.nn.ModuleList()
        self.lins.append(torch.nn.Linear(h_feats, h_feats))
        for _ in range(layer - 2):
            self.lins.append(torch.nn.Linear(h_feats, h_feats))
        self.lins.append(torch.nn.Linear(h_feats, 1))
        self.dropout = dropout
        self.res = res
        self.layer = layer
        self.alphas = nn.Parameter(torch.ones(layer + 1))

    def forward(self, x_i, x_j):
        x = x_i * x_j
        res = x * self.alphas[0]
        for i in range(self.layer - 1):
            x = self.lins[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            res += x * self.alphas[i + 1]
        x = self.lins[-1](res)
        return x.squeeze()'''

class DotPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        x = (x_i * x_j).sum(dim=-1)
        return x.squeeze()
    
class LorentzPredictor(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x_i, x_j):
        n = x_i.size(1)
        x = torch.sum(x_i[:, 0:n//2] * x_j[:, 0:n//2], dim=-1) - torch.sum(x_i[:, n//2:] * x_j[:, n//2:], dim=-1)
        return x.squeeze()

def drop_edge(g, dpe = 0.2):
    g = g.clone()
    eids = torch.randperm(g.number_of_edges())[:int(g.number_of_edges() * dpe)].to(g.device)
    g.remove_edges(eids)
    g = dgl.add_self_loop(g)
    return g

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2, dropout=0.2,
                 norm=False, tailact=False, norm_affine=True):
        super(MLP, self).__init__()
        self.lins = torch.nn.Sequential()
        self.lins.append(torch.nn.Linear(in_channels, hidden_channels))
        if norm:
            self.lins.append(nn.LayerNorm(hidden_channels, elementwise_affine=norm_affine))
        self.lins.append(nn.ReLU())
        if dropout > 0:
            self.lins.append(nn.Dropout(dropout))
        for _ in range(num_layers - 2):
            self.lins.append(torch.nn.Linear(hidden_channels, hidden_channels))
            if norm:
                self.lins.append(nn.LayerNorm(hidden_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            if dropout > 0:
                self.lins.append(nn.Dropout(dropout))
        self.lins.append(torch.nn.Linear(hidden_channels, out_channels))
        if tailact:
            self.lins.append(nn.LayerNorm(out_channels), elementwise_affine=norm_affine)
            self.lins.append(nn.ReLU())
            self.lins.append(nn.Dropout(dropout))

    def forward(self, x):
        x = self.lins(x)
        return x.squeeze()
    
class GCN(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, drop_edge=False, relu=False, linear=False, prop_step=2, dropout=0.2, residual=0, conv='GCN'):
        super(GCN, self).__init__()
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.mlp1 = MLP(in_feats, h_feats, 2, 0.2)
            self.mlp2 = MLP(h_feats, h_feats, 2, 0.2)
            self.conv1 = GINConv(self.mlp1, 'mean')
            self.conv2 = GINConv(self.mlp2, 'mean')
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        self.linear = linear
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if linear:
            self.mlps = nn.ModuleList([MLP(h_feats, h_feats, 2, dropout) for _ in range(prop_step)])

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x
    
    def forward(self, g, in_feat):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)
            h = self.conv2(g, h).flatten(1) + self.residual * ori
        return h

class GCN_multilayers(nn.Module):
        
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, drop_edge=False, relu=False, linear=False, prop_step=2, dropout=0.2, residual=0, conv='GCN'):
        super(GCN_multilayers, self).__init__()
        if conv == 'GCN':
            self.convs = nn.ModuleList([GraphConv(in_feats, h_feats)])
            for _ in range(prop_step - 1):
                self.convs.append(GraphConv(h_feats, h_feats))
        elif conv == 'SAGE':
            self.convs = nn.ModuleList([SAGEConv(in_feats, h_feats, 'mean')])
            for _ in range(prop_step - 1):
                self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
        elif conv == 'GAT':
            self.convs = nn.ModuleList([GATConv(in_feats, h_feats // 4, 4)])
            for _ in range(prop_step - 1):
                self.convs.append(GATConv(h_feats, h_feats // 4, 4))
        elif conv == 'GIN':
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, 0.2)])
            self.convs = nn.ModuleList([GINConv(self.mlps[0], 'mean')])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, 0.2))
                self.convs.append(GINConv(self.mlps[-1], 'mean'))
        self.norm = norm
        self.drop_edge = drop_edge
        self.relu = relu
        self.prop_step = prop_step
        self.residual = residual
        self.linear = linear
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
        self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        if linear:
            self.mlps = nn.ModuleList([MLP(h_feats, h_feats, 2, dropout) for _ in range(prop_step)])

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        x = self.dp(x)
        return x
    
    def forward(self, g, in_feat):
        ori = in_feat
        if self.drop_edge:
            g = drop_edge(g)
        h = self.conv1(g, in_feat).flatten(1) + self.residual * ori
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)
            h = self.conv2(g, h).flatten(1) + self.residual * ori
        return h
    
class GCN_with_feature(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, prop_step=2, dropout = 0.2, residual = 0, relu = False, linear=False, conv='GCN'):
        super(GCN_with_feature, self).__init__()
        self.conv = conv
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats)
            self.conv2 = GraphConv(h_feats, h_feats)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean')
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean')
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.mlp1 = MLP(in_feats, h_feats, 2, dropout)
            self.mlp2 = MLP(h_feats, h_feats, 2, dropout)
            self.conv1 = GINConv(self.mlp1, 'mean')
            self.conv2 = GINConv(self.mlp2, 'mean')
        self.prop_step = prop_step
        self.residual = residual
        self.relu = relu
        self.norm = norm
        self.linear = linear
        self.in_feats = in_feats
        self.h_feats = h_feats
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)
        if self.linear:
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, g, in_feat, e_feat=None):
        h = self.conv1(g, in_feat, edge_weight=e_feat).flatten(1)
        ori = h
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)
            h = self.conv2(g, h, edge_weight=e_feat).flatten(1) + self.residual * ori
        return h

class GCN_with_feature_multilayers(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, dp4norm=0, prop_step=2, dropout = 0.2, residual = 0, relu = False, linear=False, conv='GCN'):
        super(GCN_with_feature_multilayers, self).__init__()
        if conv == 'GCN':
            self.convs = nn.ModuleList([GraphConv(in_feats, h_feats)])
            for _ in range(prop_step - 1):
                self.convs.append(GraphConv(h_feats, h_feats))
        elif conv == 'SAGE':
            self.convs = nn.ModuleList([SAGEConv(in_feats, h_feats, 'mean')])
            for _ in range(prop_step - 1):
                self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
        elif conv == 'GAT':
            self.convs = nn.ModuleList([GATConv(in_feats, h_feats // 4, 4)])
            for _ in range(prop_step - 1):
                self.convs.append(GATConv(h_feats, h_feats // 4, 4))
        elif conv == 'GIN':
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            self.convs = nn.ModuleList([GINConv(self.mlps[0], 'mean')])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))
                self.convs.append(GINConv(self.mlps[-1], 'mean'))
        else:
            raise ValueError('conv type not supported')
        self.prop_step = prop_step
        self.residual = residual
        self.relu = relu
        self.norm = norm
        self.linear = linear
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)
        if self.linear:
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            for _ in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, g, in_feat, e_feat=None):
        h = self.convs[0](g, in_feat, edge_weight=e_feat).flatten(1)
        ori = h
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            if self.linear:
                h = self.mlps[i](h)
            h = self.convs[i](g, h, edge_weight=e_feat).flatten(1) + self.residual * ori
        return h

class GCN_v1(nn.Module):
    def __init__(self, in_feats, h_feats, norm=False, relu=False, prop_step=2, dropout=0.2, 
                 multilayer=False, conv='GCN', res=False, gin_aggr='sum'):
        super(GCN_v1, self).__init__()
        self.lin = nn.Linear(in_feats, h_feats) if in_feats != h_feats else nn.Identity()
        self.multilayer = multilayer
        self.conv = conv
        if multilayer:
            self.convs = nn.ModuleList()
            for _ in range(prop_step):
                if conv == 'GCN':
                    self.convs.append(GraphConv(h_feats, h_feats))
                elif conv == 'SAGE':
                    self.convs.append(SAGEConv(h_feats, h_feats, 'mean'))
                elif conv == 'GAT':
                    self.convs.append(GATConv(h_feats, h_feats // 4, 4))
                elif conv == 'GIN':
                    self.convs.append(GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm), gin_aggr))
        else:
            if conv == 'GCN':
                self.conv = GraphConv(h_feats, h_feats)
            elif conv == 'SAGE':
                self.conv = SAGEConv(h_feats, h_feats, 'mean')
            elif conv == 'GAT':
                self.conv = GATConv(h_feats, h_feats // 4, 4)
            elif conv == 'GIN':
                self.conv = GINConv(MLP(h_feats, h_feats, h_feats, 2, dropout, norm), gin_aggr)
        self.norm = norm
        self.relu = relu
        self.prop_step = prop_step
        if norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.res = res

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x
    
    def forward(self, g, in_feat, e_feat=None):
        h = self.lin(in_feat)
        h = self._apply_norm_and_activation(h, 0)
        ori = h
        for i in range(self.prop_step):
            if i != 0:
                if self.res:
                    h = h + ori
                h = self._apply_norm_and_activation(h, i)
            if self.multilayer:
                h = self.convs[i](g, h, edge_weight=e_feat).flatten(1)
            else:
                h = self.conv(g, h, edge_weight=e_feat).flatten(1)
        return h


'''class MultiheadLightGCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, alpha = 0.5, num_heads = 1, exp = False, relu = False):
        super(MultiheadLightGCN, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, weight=True, bias=False)
        self.conv2 = GraphConv(h_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.dp = dropout
        self.relu = relu
        self.num_heads = num_heads
        self.alpha = alpha
        if exp:
            self.alphas = nn.ParameterList([nn.Parameter(alpha ** torch.arange(prop_step, dtype=torch.float32)) for _ in range(num_heads)])
        else:
            self.alphas = nn.ParameterList([nn.Parameter(torch.ones(prop_step, dtype=torch.float32)) for _ in range(num_heads)])

    def forward(self, g, in_feat, e_feat=None):

        h = self.conv1(g, in_feat, edge_weight=e_feat).flatten(1)

        hs = [h]
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
                h = F.dropout(h, p=self.dp, training=self.training)
            h = self.conv2(g, h, edge_weight=e_feat).flatten(1)
            hs.append(h)

        res = torch.zeros_like(hs[0])
        for i in range(self.num_heads):
            for j in range(self.prop_step):
                res += hs[j] * self.alphas[i][j]
        res /= self.num_heads
        return res'''
    
class LightGCN(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, alpha = 0.5, exp = False, relu = False, norm=False, conv='GCN'):
        super(LightGCN, self).__init__()
        self.conv = conv
        if conv == 'GCN':
            self.conv1 = GraphConv(in_feats, h_feats, weight=True, bias=False)
            self.conv2 = GraphConv(h_feats, h_feats, weight=False, bias=False)
        elif conv == 'SAGE':
            self.conv1 = SAGEConv(in_feats, h_feats, 'mean', bias=False)
            self.conv2 = SAGEConv(h_feats, h_feats, 'mean', bias=False)
        elif conv == 'GAT':
            self.conv1 = GATConv(in_feats, h_feats // 4, 4)
            self.conv2 = GATConv(h_feats, h_feats // 4, 4)
        elif conv == 'GIN':
            self.mlps = nn.ModuleList([MLP(in_feats, h_feats, 2, dropout)])
            self.convs = nn.ModuleList([GINConv(self.mlps[0], 'sum')])
            for i in range(prop_step - 1):
                self.mlps.append(MLP(h_feats, h_feats, 2, dropout))
                self.convs.append(GINConv(self.mlps[i + 1], 'sum'))
        self.prop_step = prop_step
        self.relu = relu
        self.alpha = alpha
        if exp:
            self.alphas = nn.Parameter(alpha ** torch.arange(prop_step))
        else:
            self.alphas = nn.Parameter(torch.ones(prop_step))
        self.norm = norm
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(h_feats) for _ in range(prop_step)])
            self.dp = nn.Dropout(dropout)

    def _apply_norm_and_activation(self, x, i):
        if self.norm:
            x = self.norms[i](x)
        if self.relu:
            x = F.relu(x)
        if self.norm:
            x = self.dp(x)
        return x

    def forward(self, g, in_feat, e_feat=None):
        if self.conv == 'GIN':
            alpha = F.softmax(self.alphas, dim=0)
            h = self.convs[0](g, in_feat, edge_weight=e_feat).flatten(1)
            res = h * alpha[0]
            for i in range(1, self.prop_step):
                h = self._apply_norm_and_activation(h, i)
                h = self.convs[i](g, h, edge_weight=e_feat).flatten(1)
                res += h * alpha[i]
            return res
        alpha = F.softmax(self.alphas, dim=0)
        h = self.conv1(g, in_feat, edge_weight=e_feat).flatten(1)
        res = h * alpha[0]
        for i in range(1, self.prop_step):
            h = self._apply_norm_and_activation(h, i)
            h = self.conv2(g, h, edge_weight=e_feat).flatten(1)
            res += h * alpha[i]
        return res
    
'''class LightGCN1(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, alpha = 0.5, exp = False, relu = False):
        super(LightGCN1, self).__init__()
        self.conv = GraphConv(in_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.dp = dropout
        self.relu = relu
        self.alpha = alpha
        if exp:
            self.alphas = nn.Parameter(alpha ** torch.arange(prop_step + 1))
        else:
            self.alphas = nn.Parameter(torch.ones(prop_step + 1))

    def forward(self, g, in_feat, e_feat=None):
        h = in_feat
        # alphas = F.softmax(self.alphas, dim=0)
        res = h * self.alphas[0]
        for i in range(self.prop_step):
            if self.relu:
                h = F.relu(h)
                h = F.dropout(h, p=self.dp, training=self.training)
            h = self.conv(g, h, edge_weight=e_feat).flatten(1)
            res += h * self.alphas[i + 1]
        return res

class LightGCN_res(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step=2, dropout = 0.2, relu = False):
        super(LightGCN_res, self).__init__()
        self.conv1 = GraphConv(in_feats, h_feats, weight=True, bias=False)
        self.conv2 = GraphConv(h_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.dp = dropout
        self.relu = relu

    def forward(self, g, in_feat, e_feat=None):
        h = self.conv1(g, in_feat, edge_weight=e_feat).flatten(1)
        for i in range(1, self.prop_step):
            if self.relu:
                h = F.relu(h)
                h = F.dropout(h, p=self.dp, training=self.training)
            h = self.conv2(g, h, edge_weight=e_feat) + h
        return h

class GCN_with_MLP(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.2, prop_step = 2, relu = False):
        super(GCN_with_MLP, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv = GraphConv(h_feats, h_feats)
        self.prop_step = prop_step
        self.relu = relu

    def forward(self, g, in_feat):
        h = self.mlp(in_feat)
        for i in range(self.prop_step):
            if self.relu:
                h = F.relu(h)
            h = self.conv(g, h)
        return h
    
class GCN_no_para(nn.Module):
    def __init__(self, in_feats, h_feats, dropout = 0.2, prop_step = 2, relu = False):
        super(GCN_no_para, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.conv = GraphConv(h_feats, h_feats, weight=False, bias=False)
        self.prop_step = prop_step
        self.relu = relu

    def forward(self, g, in_feat):
        h = self.mlp(in_feat)
        for i in range(self.prop_step):
            if self.relu:
                h = F.relu(h)
            h = self.conv(g, h)
        return h'''
    
class PureGCN(nn.Module):
    def __init__(self, input_dim, num_layers=2, hidden=256, dp=0, relu=False, norm=False, res=False):
        super().__init__()
        self.lin = nn.Linear(input_dim, hidden)
        self.conv = GraphConv(hidden, hidden, weight=False, bias=False)
        self.num_layers = num_layers
        self.dp = dp
        self.norm = norm
        self.res = res
        self.relu = relu
        if self.norm:
            self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(num_layers)])

    def forward(self, adj_t, x, e_feat=None):
        x = self.lin(x)
        ori = x
        for i in range(self.num_layers):
            if i != 0 and self.res:
                x = x + ori
            if self.norm:
                x = self.norms[i](x)
            if self.relu:
                x = F.relu(x)
            if self.dp > 0:
                x = F.dropout(x, p=self.dp, training=self.training)
            x = self.conv(adj_t, x, edge_weight=e_feat)
        return x

class PureGCN_no_para(nn.Module):
    def __init__(self, input_dim, num_layers=2, relu=False, norm=False, res=False):
        super().__init__()
        self.conv = GraphConv(input_dim, input_dim, weight=False, bias=False)
        self.num_layers = num_layers
        self.norm = norm
        self.res = res
        self.relu = relu

    def forward(self, adj_t, x, e_feat=None):
        ori = x
        for i in range(self.num_layers):
            if i != 0 and self.res:
                x = x + ori
            if self.norm:
                x = F.layer_norm(x, x.shape[1:])
            if self.relu:
                x = F.relu(x)
            x = self.conv(adj_t, x, edge_weight=e_feat)
        return x       