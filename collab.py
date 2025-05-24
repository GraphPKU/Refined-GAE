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
from model import GCN_with_feature, DotPredictor, LightGCN, Hadamard_MLPPredictor, GCN_with_feature_multilayers
import time
import wandb

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-collab', choices=['ogbl-collab'], type=str)
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
    parser.add_argument("--model", default='GCN', type=str)
    parser.add_argument("--maskinput", action='store_true', default=False)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float)
    parser.add_argument("--dpe", default=0, type=float)
    parser.add_argument("--drop_edge", action='store_true', default=False)
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank'], type=str)
    parser.add_argument("--residual", default=0, type=float)
    parser.add_argument("--mlp_layers", default=2, type=int)
    parser.add_argument("--pred", default='mlp', type=str)
    parser.add_argument("--res", action='store_true', default=False)
    parser.add_argument("--conv", default='GCN', type=str)
    parser.add_argument("--use_valid_as_input", action='store_true', default=False)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--exp', action='store_true', default=False)
    parser.add_argument('--scale', action='store_true', default=False)
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--init', default='orthogonal', type=str)

    args = parser.parse_args()
    return args

args = parse()
print(args)
wandb.init(project='Refined-GAE', config=args)

torch.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
dgl.seed(args.seed)

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

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred, embedding=None):
    model.train()
    pred.train()

    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    if args.maskinput:
        mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool)
    sample_time = 0
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        if args.maskinput:
            mask[edge_index] = 0
            tei = train_pos_edge[mask]
            tei = tei.unique(dim=0)
            src, dst = tei.t()
            re_tei = torch.stack((dst, src), dim=0).t()
            tei = torch.cat((tei, re_tei), dim=0)
            g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())
            g_mask = dgl.add_self_loop(g_mask)
            edge_weight = torch.ones(g_mask.number_of_edges(), dtype=torch.float32).to(device)
            h = model(g_mask, xemb, edge_weight)
            mask[edge_index] = 1
        else:
            h = model(g, xemb, g.edata['weight'])

        pos_edge = train_pos_edge[edge_index]
        st = time.time()
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        sample_time += time.time() - st
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
            pos_loss = -F.logsigmoid(pos_score).mean()
            neg_loss = -F.logsigmoid(-neg_score).mean()
            loss = pos_loss + neg_loss
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
        torch.nn.utils.clip_grad_norm_(pred.parameters(), args.clip_norm)
        if embedding is not None:
            torch.nn.utils.clip_grad_norm_(embedding.parameters(), args.clip_norm)
        optimizer.step()
        total_loss += loss.item()
        print(f"Sample time: {sample_time:.4f}", flush=True)

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, pred, embedding=None):
    model.eval()
    pred.eval()

    with torch.no_grad():
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        h = model(g, xemb, g.edata['weight'])
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
        for k in [20, 50, 100]:
            results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
    return results

def eval(model, g, pos_train_edge, pos_valid_edge, neg_valid_edge, pred, embedding=None):
    model.eval()
    pred.eval()

    with torch.no_grad():
        xemb = torch.cat((embedding.weight, g.ndata['feat']), dim=1) if embedding is not None else g.ndata['feat']
        h = model(g, xemb, g.edata['weight'])
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
        valid_results = {}
        for k in [20, 50, 100]:
            valid_results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
        pos_score = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_train_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score.append(pos_pred)
        pos_score = torch.cat(pos_score, dim=0)
        train_results = {}
        for k in [20, 50, 100]:
            train_results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
    return valid_results, train_results

# Load the dataset
dataset = DglLinkPropPredDataset(name=args.dataset)
split_edge = dataset.get_edge_split()

device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

graph = dataset[0]

if args.dataset == 'ogbl-collab':
    selected_year_index = torch.reshape(
        (split_edge['train']['year'] >= 2011).nonzero(as_tuple=False), (-1,))
    split_edge['train']['edge'] = split_edge['train']['edge'][selected_year_index]
    split_edge['train']['weight'] = split_edge['train']['weight'][selected_year_index]
    split_edge['train']['year'] = split_edge['train']['year'][selected_year_index]
    train_edge_index = split_edge['train']['edge'].t()

    if args.use_valid_as_input:
        u, v = split_edge['valid']['edge'].t()
        re_vei = torch.stack((v, u), dim=0)

        u, v = split_edge['train']['edge'].t()
        re_tei = torch.stack((v, u), dim=0)

        train_edge_index = torch.cat((train_edge_index, re_tei, re_vei, split_edge['valid']['edge'].t()), dim=1)
        train_edge_weight = torch.cat((split_edge['train']['weight'], split_edge['train']['weight'], split_edge['valid']['weight'], split_edge['valid']['weight']), dim=0)

        feat = graph.ndata['feat']
        graph = dgl.graph((train_edge_index[0], train_edge_index[1]), num_nodes=graph.num_nodes())
        graph.ndata['feat'] = feat
        graph.edata['weight'] = train_edge_weight.to(torch.float32)
        split_edge['train']['edge'] = torch.cat((split_edge['train']['edge'], split_edge['valid']['edge']), dim=0)
        graph = dgl.add_self_loop(graph, fill_data=0)
        graph = graph.to(device)
    else:
        feat = graph.ndata['feat']
        graph = dgl.graph((train_edge_index[0], train_edge_index[1]), num_nodes=graph.num_nodes())
        graph.ndata['feat'] = feat
        graph = dgl.add_self_loop(graph)
        graph = dgl.to_bidirected(graph, copy_ndata=True)
        graph.edata['weight'] = torch.ones(graph.number_of_edges(), dtype=torch.float32)
        graph = graph.to(device)

train_pos_edge = split_edge['train']['edge'].to(device)
valid_pos_edge = split_edge['valid']['edge'].to(device)
valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
test_pos_edge = split_edge['test']['edge'].to(device)
test_neg_edge = split_edge['test']['edge_neg'].to(device)

if args.emb_hidden > 0:
    embedding = torch.nn.Embedding(graph.num_nodes(), args.emb_hidden).to(device)
    if args.init == 'orthogonal':
        torch.nn.init.orthogonal_(embedding.weight)
    elif args.init == 'ones':
        torch.nn.init.ones_(embedding.weight)
    elif args.init == 'random':
        torch.nn.init.uniform_(embedding.weight)
else:
    embedding = None

# Create negative samples for training
neg_sampler = GlobalUniform(args.num_neg)

if args.pred == 'dot':
    pred = DotPredictor().to(device)
elif args.pred == 'mlp':
    pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.res, args.norm, args.scale).to(device)
else:
    raise NotImplementedError

input_dim = graph.ndata['feat'].shape[1] + args.emb_hidden if embedding is not None else graph.ndata['feat'].shape[1]

if args.model == 'GCN':
    model = GCN_with_feature(input_dim, args.hidden, args.norm, args.dropout, args.prop_step, args.dropout, args.residual, args.relu, args.linear, args.conv).to(device)
elif args.model == 'LightGCN':
    model = LightGCN(input_dim, args.hidden, args.prop_step, args.dropout, args.alpha, args.exp, args.relu).to(device)
elif args.model == 'GCN_multilayer':
    model = GCN_with_feature_multilayers(input_dim, args.hidden, args.norm, args.dropout, args.prop_step, args.dropout, args.residual, args.relu, args.linear).to(device)
else:
    raise NotImplementedError


parameter = itertools.chain(model.parameters(), pred.parameters())
if embedding is not None:
    parameter = itertools.chain(parameter, embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)
evaluator = Evaluator(name=args.dataset)

best_val = 0
final_test_result = None
best_epoch = 0

losses = []
valid_list = []
test_list = []

if embedding is not None:
    print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters()) + sum(p.numel() for p in embedding.parameters())}')
else:
    print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters())}')

tot_time = 0
for epoch in range(args.epochs):
    st = time.time()
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, embedding)
    print(f"Epoch {epoch}, Time: {time.time()-st:.4f}", flush=True)
    tot_time += time.time() - st
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results, train_results = eval(model, graph, train_pos_edge, valid_pos_edge, valid_neg_edge, pred, embedding)
    valid_list.append(valid_results[args.metric])
    for k, v in valid_results.items():
        print(f'Validation {k}: {v:.4f}')
    for k, v in train_results.items():
        print(f'Train {k}: {v:.4f}')
    if not args.use_valid_as_input:
        graph_t = graph.clone()
        u, v = valid_pos_edge.t()
        graph_t.add_edges(u, v)
        graph_t.add_edges(v, u)
        graph_t.edata['weight'] = torch.ones(graph_t.number_of_edges(), dtype=torch.float32, device=device)
    else:
        graph_t = graph
    test_results = test(model, graph_t, test_pos_edge, test_neg_edge, pred, embedding)
    test_list.append(test_results[args.metric])
    for k, v in test_results.items():
        print(f'Test {k}: {v:.4f}')

    if valid_results[args.metric] >= best_val:
        best_val = valid_results[args.metric]
        best_epoch = epoch
        final_test_result = test_results
    if epoch - best_epoch >= 200:
        break
    print(f"Epoch {epoch}, Loss: {loss:.4f}, Train {args.metric}: {train_results[args.metric]:.4f}, Valid {args.metric}: {valid_results[args.metric]:.4f}, Test {args.metric}: {test_results[args.metric]:.4f}")
    wandb.log({'loss': loss, 'train_hit': train_results[args.metric], 'valid_hit': valid_results[args.metric], 'test_hit': test_results[args.metric]})

print(f'total time: {tot_time:.4f}')
print(f"Test hit: {final_test_result[args.metric]:.4f}")
wandb.log({'final_test_hit': final_test_result[args.metric]})
