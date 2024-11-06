import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset, Evaluator
from dgl.dataloading.negative_sampler import GlobalUniform, PerSourceUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
from loss import auc_loss, hinge_auc_loss, log_rank_loss
from model import Hadamard_MLPPredictor, GCN, GCN_with_MLP, GCN_no_para, GCN_with_feature

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-ddi', choices=['ogbl-ddi', 'ogbl-ppa'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--hidden", default=32, type=int)
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
    parser.add_argument("--model", default='GCN', choices=['GCN', 'GCN_with_feature', 'GCN_with_MLP', 'GCN_no_para'], type=str)
    parser.add_argument("--maskinput", action='store_true', default=False)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float)
    parser.add_argument("--dpe", default=0, type=float)
    parser.add_argument("--drop_edge", action='store_true', default=False)
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank'], type=str)
    parser.add_argument("--residual", default=0, type=float)
    parser.add_argument("--mlp_layers", default=2, type=int)

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

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred, feat):
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
            h = model(g_mask, feat())
            mask[edge_index] = 1
        else:
            h = model(g, feat())

        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0)
        neg_train_edge = neg_train_edge.t()
        neg_edge = neg_train_edge

        pos_score = pred(h[pos_edge[:,0]], h[pos_edge[:,1]])
        neg_score = pred(h[neg_edge[:,0]], h[neg_edge[:,1]])
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
        if args.dataset == 'ogbl-ddi':
            torch.nn.utils.clip_grad_norm_(feat.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, evaluator, pred, feat):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, feat())
        dataloader = DataLoader(range(pos_test_edge.size(0)), args.batch_size)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_test_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_test_edge.size(0)), args.batch_size)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_test_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

def eval(model, g, pos_valid_edge, neg_valid_edge, evaluator, pred, feat):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, feat())
        dataloader = DataLoader(range(pos_valid_edge.size(0)), args.batch_size)
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_valid_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        dataloader = DataLoader(range(neg_valid_edge.size(0)), args.batch_size)
        neg_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            neg_edge = neg_valid_edge[edge_index]
            neg_pred = pred(h[neg_edge[:, 0]], h[neg_edge[:, 1]])
            neg_score.append(neg_pred)
        neg_score = torch.cat(neg_score, dim=0)
        results = {}
        results[args.metric] = evaluator.eval({
            'y_pred_pos': pos_score,
            'y_pred_neg': neg_score,
        })[args.metric]
    return results

class feat_model(nn.Module):
    def __init__(self, in_feats):
        super(feat_model, self).__init__()
        self.feat = nn.Parameter(in_feats, requires_grad=True)

    def forward(self):
        return self.feat

# Load the dataset 
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]
graph = dgl.add_self_loop(graph).to(device)
#graph = dgl.to_bidirected(graph, copy_ndata=True).to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers).to(device)

embedding = torch.nn.Embedding(graph.num_nodes(), args.hidden)
torch.nn.init.orthogonal_(embedding.weight)
feat = embedding.weight + 1
feat = feat_model(feat).to(device)

if args.model == 'GCN':
    model = GCN(embedding.weight.shape[1], args.hidden, args.norm, args.dp4norm, args.drop_edge, args.relu, args.prop_step, args.residual).to(device)
elif args.model == 'GCN_with_MLP':
    model = GCN_with_MLP(embedding.weight.shape[1], args.hidden, args.norm, args.dp4norm, args.drop_edge, args.relu, args.prop_step, args.residual).to(device)
elif args.model == 'GCN_no_para':
    model = GCN_no_para(embedding.weight.shape[1], args.hidden, args.prop_step, args.dropout, args.residual).to(device)
elif args.model == 'GCN_with_feature':
    model = GCN_with_feature(embedding.weight.shape[1], args.hidden, args.norm, args.dp4norm, args.drop_edge, args.relu, args.prop_step, args.residual).to(device)
else:
    raise NotImplementedError

parameter = itertools.chain(model.parameters(), pred.parameters(), feat.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0

losses = []
valid_list = []
test_list = []

for epoch in range(args.epochs):
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, feat)
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results = eval(model, graph, valid_pos_edge, valid_neg_edge, evaluator, pred, feat)
    valid_list.append(valid_results[args.metric])
    test_results = test(model, graph, test_pos_edge, test_neg_edge, evaluator, pred, feat)
    test_list.append(test_results[args.metric])
    if valid_results[args.metric] > best_val:
        if args.dataset == 'ogbl-ddi' and epoch > 300:
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
        elif args.dataset != 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
    if args.dataset == 'ogbl-ppa':
        if best_epoch + 100 < epoch:
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