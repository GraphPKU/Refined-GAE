import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
from loss import auc_loss, hinge_auc_loss, log_rank_loss
from model import GCN_with_feature, Hadamard_MLPPredictor
from dgl.nn import GraphConv

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-collab', choices=['ogbl-collab', 'ogbl-citation2'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--hidden", default=32, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=1, type=int)
    parser.add_argument("--epochs", default=50, type=int)
    parser.add_argument("--interval", default=50, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=False)
    parser.add_argument("--metric", default='hits@20', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--relu", action='store_true', default=False)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", default='GCN', choices=['GCN', 'GCN_with_MLP', 'GCN_no_para'], type=str)
    parser.add_argument("--maskinput", action='store_true', default=False)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float)
    parser.add_argument("--dpe", default=0, type=float)
    parser.add_argument("--drop_edge", action='store_true', default=False)
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank'], type=str)
    parser.add_argument("--decay", default=0.01, type=float)

    args = parser.parse_args()
    return args

args = parse()
print(args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dgl.seed(args.seed)

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    if args.maskinput:
        mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool)
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        if args.maskinput:
            mask[edge_index] = 0
            tei = train_pos_edge[mask]
            src, dst = tei.t()
            re_tei = torch.stack((dst, src), dim=0).t()
            tei = torch.cat((tei, re_tei), dim=0)
            g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())
            g_mask = dgl.add_self_loop(g_mask)
            h = model(g_mask, g.ndata['feat_2'], g.ndata['feat'])
            mask[edge_index] = 1
        else:
            h = model(g, g.ndata['feat_2'], g.ndata['feat'])

        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge
        pos_score = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
        neg_score = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
        if args.loss == 'auc':
            loss = auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'hauc':
            loss = hinge_auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'rank':
            loss = log_rank_loss(pos_score, neg_score, args.num_neg)
        else:
            loss = F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) + F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(g.ndata['feat_2'], 1.0)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, evaluator, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat_2'], g.ndata['feat'])
        dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size, shuffle=True)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_test_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size, shuffle=True)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_test_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        if args.dataset == 'ogbl-citation2':
            neg_score = neg_score.view(-1, 1000)
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })['mrr_list'].mean().item()
        else:
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })[args.metric]
    return results

def eval(model, g, pos_valid_edge, neg_valid_edge, evaluator, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat_2'], g.ndata['feat'])
        dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size, shuffle=True)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_valid_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size, shuffle=True)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_valid_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        if args.dataset == 'ogbl-citation2':
            neg_score = neg_score.view(-1, 1000)
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })['mrr_list'].mean().item()
        else:
            results = {}
            results[args.metric] = evaluator.eval({
                'y_pred_pos': pos_score,
                'y_pred_neg': neg_score,
            })[args.metric]
    return results

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

if args.dataset =="ogbl-citation2":
    for name in ['train','valid','test']:
        u=split_edge[name]["source_node"]
        v=split_edge[name]["target_node"]
        split_edge[name]['edge']=torch.stack((u,v),dim=0).t()
    for name in ['valid','test']:
        u=split_edge[name]["source_node"].repeat(1, 1000).view(-1)
        v=split_edge[name]["target_node_neg"].view(-1)
        split_edge[name]['edge_neg']=torch.stack((u,v),dim=0).t()

train_pos_edge = split_edge['train']['edge']
valid_pos_edge = split_edge['valid']['edge']
valid_neg_edge = split_edge['valid']['edge_neg']
test_pos_edge = split_edge['test']['edge']
test_neg_edge = split_edge['test']['edge_neg']

if args.dataset == 'ogbl-collab':
    num_node = dataset[0].num_nodes()
    year_diff = 2017 - split_edge['train']['year']
    weights = torch.exp(-args.decay * year_diff.float())
    src, dst = train_pos_edge.t()
    re_train_pos_edge = torch.stack((dst, src), dim=0).t()
    self_loop = torch.stack((torch.arange(num_node), torch.arange(num_node)), dim=0).t()
    train_pos_edge_bi = torch.cat((train_pos_edge, re_train_pos_edge, self_loop), dim=0)
    graph = dgl.graph((train_pos_edge_bi[:, 0], train_pos_edge_bi[:, 1]), num_nodes=num_node)
    graph.edata['weight'] = torch.cat((weights, weights, torch.ones(num_node)))
    src, dst = valid_pos_edge.t()
    re_valid_pos_edge = torch.stack((dst, src), dim=0).t()
    valid_pos_edge_bi = torch.cat((valid_pos_edge, re_valid_pos_edge), dim=0)
    edge = torch.cat((train_pos_edge_bi, valid_pos_edge_bi), dim=0)
    graph_t = dgl.graph((edge[:, 0], edge[:, 1]), num_nodes=num_node)
    graph_t.edata['weight'] = torch.cat((weights, weights, torch.ones(num_node), torch.ones(valid_pos_edge_bi.size(0))))
    graph.ndata['feat'] = dataset[0].ndata['feat']
    graph_t.ndata['feat'] = dataset[0].ndata['feat']
    graph = graph.to(device)
    graph_t = graph_t.to(device)

    print(train_pos_edge.shape)

train_pos_edge = train_pos_edge.to(device)
valid_pos_edge = valid_pos_edge.to(device)
valid_neg_edge = valid_neg_edge.to(device)
test_pos_edge = test_pos_edge.to(device)
test_neg_edge = test_neg_edge.to(device)

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

pred = Hadamard_MLPPredictor(args.hidden, args.dropout).to(device)

embedding = torch.nn.Embedding(graph.num_nodes(), args.hidden).to(device)
torch.nn.init.orthogonal_(embedding.weight)
graph.ndata['feat_2'] = embedding.weight
graph_t.ndata['feat_2'] = embedding.weight

class GCN_collab(nn.Module):
    def __init__(self, in_feats, h_feats, prop_step = 2, dropout = 0.2):
        super(GCN_collab, self).__init__()
        self.conv1 = GraphConv(h_feats, h_feats)
        self.conv2 = GraphConv(h_feats, h_feats)
        self.conv1_feat = GraphConv(in_feats, h_feats)
        self.conv2_feat = GraphConv(h_feats, h_feats)
        self.mlp = nn.Sequential(
            nn.Linear(2 * h_feats, h_feats),
            nn.BatchNorm1d(h_feats),
            nn.Dropout(dropout),
            nn.ReLU(),
            nn.Linear(h_feats, h_feats)
        )
        self.prop_step = prop_step

    def forward(self, g, in_feat, in_feat2):
        h = self.conv1(g, in_feat, edge_weight=g.edata['weight'])
        h2 = self.conv1_feat(g, in_feat2, edge_weight=g.edata['weight'])
        h2 = F.relu(h2)
        for _ in range(1, self.prop_step):
            h = self.conv2(g, h, edge_weight=g.edata['weight'])
            h2 = self.conv2_feat(g, h2, edge_weight=g.edata['weight'])
            h2 = F.relu(h2)
        h = self.mlp(torch.cat([h, h2], dim=1))
        return h

model = GCN_collab(graph.ndata['feat'].shape[1], args.hidden, args.prop_step, args.dropout).to(device)

parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0

losses = []
valid_list = []
test_list = []


for epoch in range(args.epochs):
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred)
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results = eval(model, graph, valid_pos_edge, valid_neg_edge, evaluator, pred)
    valid_list.append(valid_results[args.metric])
    test_results = test(model, graph_t, test_pos_edge, test_neg_edge, evaluator, pred)
    test_list.append(test_results[args.metric])
    if valid_results[args.metric] > best_val:
        best_val = valid_results[args.metric]
        best_epoch = epoch
        final_test_result = test_results
    if args.dataset == 'ogbl-collab':
        if epoch - best_epoch >= 200:
            break
    else:
        if epoch - best_epoch >= 100:
            break
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Validation hit: {valid_results[args.metric]:.4f}, Test hit: {test_results[args.metric]:.4f}")

print(f"Test hit: {final_test_result[args.metric]:.4f}")

import matplotlib.pyplot as plt

plt.figure()
plt.plot(range(len(losses)), losses, label='loss')
plt.plot(range(len(losses)), valid_list, label='valid')
plt.plot(range(len(losses)), test_list, label='test')
plt.xlabel('epoch')
plt.ylabel('metric')
plt.legend()
plt.savefig("plot-" + args.dataset + str(args.gpu) + ".png")