import itertools
import os
os.environ['DGLBACKEND'] = 'pytorch'

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from ogb.linkproppred import DglLinkPropPredDataset
from dgl.dataloading.negative_sampler import GlobalUniform
from torch.utils.data import DataLoader
import tqdm
import argparse
from loss import auc_loss, hinge_auc_loss, log_rank_loss
from model import Hadamard_MLPPredictor, GCN, GCN_v1, DotPredictor, LorentzPredictor
import wandb

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

def parse():
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='ogbl-ddi', choices=['ogbl-ddi', 'ogbl-ppa', 'ogbl-collab'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--emb_dim", default=32, type=int)
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
    parser.add_argument("--model", default='GCN', choices=['GCN', 'GCN_with_feature', 'GCN_with_MLP', 'GCN_v1', 'LightGCN', 'LightGCN_res', 'MultiheadLightGCN'], type=str)
    parser.add_argument("--maskinput", action='store_true', default=False)
    parser.add_argument("--norm", action='store_true', default=False)
    parser.add_argument("--mlp_norm", action='store_true', default=False)
    parser.add_argument("--dp4norm", default=0, type=float)
    parser.add_argument("--dpe", default=0, type=float)
    parser.add_argument("--drop_edge", action='store_true', default=False)
    parser.add_argument("--loss", default='bce', choices=['bce', 'auc', 'hauc', 'rank'], type=str)
    parser.add_argument("--residual", default=0, type=float)
    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--mlp_res', action='store_true', default=False)
    parser.add_argument('--conv', default='GCN', choices=['GCN', 'GAT', 'GIN', 'SAGE'], type=str)
    parser.add_argument('--pred', default='Hadamard', choices=['Hadamard', 'Dot', 'Lorentz', 'AttMLP', 'Block'], type=str)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--exp', action='store_true', default=False)
    parser.add_argument('--force_orthogonal', action='store_true', default=False)
    parser.add_argument('--init', default='orthogonal', choices=['orthogonal', 'uniform', 'ones'], type=str)
    parser.add_argument('--linear', action='store_true', default=False)
    parser.add_argument('--clip_norm', default=1.0, type=float)
    parser.add_argument('--gin_aggr', default='sum', choices=['mean', 'sum'], type=str)
    parser.add_argument('--multilayer', action='store_true', default=False)
    parser.add_argument('--res', action='store_true', default=False)

    args = parser.parse_args()
    return args

args = parse()
print(args)

wandb.init(project='Refined-GAE', config=args)

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
            h = model(g_mask, g.ndata['feat'])
            mask[edge_index] = 1
        else:
            h = model(g, g.ndata['feat'])

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
        if args.force_orthogonal:
            loss += 1e-8 * torch.norm(h @ h.t() - torch.diag(torch.diag(h @ h.t())), p='fro')
        
        optimizer.zero_grad()
        loss.backward()
        if args.dataset == 'ogbl-ddi' or args.dataset == 'ogbl-collab':
            torch.nn.utils.clip_grad_norm_(g.ndata['feat'], args.clip_norm)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_norm)
            torch.nn.utils.clip_grad_norm_(pred.parameters(), args.clip_norm)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
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

def eval(model, g, pos_train_edge, pos_valid_edge, neg_valid_edge, pred):
    model.eval()
    pred.eval()

    with torch.no_grad():
        h = model(g, g.ndata['feat'])
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

def plot_dot_product_dist(x):
    dot_products = x @ x.t()
    dot_products = dot_products.detach().cpu().numpy()
    import matplotlib.pyplot as plt
    plt.hist(dot_products.flatten(), bins=100)
    plt.xlabel('Dot Product')
    plt.ylabel('Frequency')
    plt.title('Dot Product Distribution')
    plt.show()
    plt.savefig('dot_product_distribution.png')

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

if args.pred == 'Hadamard':
    pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.mlp_res, args.mlp_norm).to(device)
elif args.pred == 'Dot':
    pred = DotPredictor().to(device)
elif args.pred == 'Lorentz':
    pred = LorentzPredictor().to(device)
else:
    raise NotImplementedError

embedding = torch.nn.Embedding(graph.num_nodes(), args.emb_dim).to(device)
if args.init == 'uniform':
    torch.nn.init.uniform_(embedding.weight)
elif args.init == 'ones':
    torch.nn.init.ones_(embedding.weight)
elif args.init == 'orthogonal':
    torch.nn.init.orthogonal_(embedding.weight)
graph.ndata['feat'] = embedding.weight

if args.model == 'GCN':
    model = GCN(graph.ndata['feat'].shape[1], args.hidden, args.norm, args.dp4norm, args.drop_edge, args.relu, args.linear, args.prop_step, args.dropout, args.residual, args.conv).to(device)
elif args.model == 'GCN_v1':
    model = GCN_v1(graph.ndata['feat'].shape[1], args.hidden, args.norm, args.relu, args.prop_step, args.dropout, args.multilayer, args.conv, args.res, args.gin_aggr).to(device)
else:
    raise NotImplementedError

parameter = itertools.chain(model.parameters(), pred.parameters(), embedding.parameters())
optimizer = torch.optim.Adam(parameter, lr=args.lr)

best_val = 0
final_test_result = None
best_epoch = 0

losses = []
valid_list = []
test_list = []

print(f'number of parameters: {sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in pred.parameters()) + sum(p.numel() for p in embedding.parameters())}')

for epoch in range(args.epochs):
    loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred)
    losses.append(loss)
    if epoch % args.interval == 0 and args.step_lr_decay:
        adjustlr(optimizer, epoch / args.epochs, args.lr)
    valid_results, train_results = eval(model, graph, train_pos_edge, valid_pos_edge, valid_neg_edge, pred)
    valid_list.append(valid_results[args.metric])
    for k, v in valid_results.items():
        print(f'Validation {k}: {v:.4f}')
    for k, v in train_results.items():
        print(f'Train {k}: {v:.4f}')
    if args.dataset == 'ogbl-collab':
        graph_t = graph.clone()
        u, v = valid_pos_edge.t()
        graph_t.add_edges(u, v)
        graph_t.add_edges(v, u)
    else:
        graph_t = graph
    test_results = test(model, graph_t, test_pos_edge, test_neg_edge, pred)
    test_list.append(test_results[args.metric])
    for k, v in test_results.items():
        print(f'Test {k}: {v:.4f}')

    if args.dataset == 'ogbl-ppa':
        if best_epoch + 200 < epoch:
            break
    if args.dataset == 'ogbl-ddi':
        if train_results[args.metric] > 0.90:
            break

    if valid_results[args.metric] > best_val:
        if args.dataset == 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
        elif args.dataset != 'ogbl-ddi':
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results

    print(f"Epoch {epoch}, Loss: {loss:.4f}, Train hit: {train_results[args.metric]:.4f}, Valid hit: {valid_results[args.metric]:.4f}, Test hit: {test_results[args.metric]:.4f}")
    wandb.log({'loss': loss, 'train_hit': train_results[args.metric], 'valid_hit': valid_results[args.metric], 'test_hit': test_results[args.metric]})

# plot_dot_product_dist(graph.ndata['feat'])
print(f"Test hit: {final_test_result[args.metric]:.4f}")
wandb.log({'final_test_hit': final_test_result[args.metric]})
