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
from model import Hadamard_MLPPredictor, GCN_v1, DotPredictor, LightGCN, PureGCN
import time
import numpy as np
import scipy.sparse as sp
import wandb
import math

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='Cora', choices=['Cora', 'CiteSeer', 'PubMed'], type=str)
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--prop_step", default=8, type=int)
    parser.add_argument("--emb_hidden", default=0, type=int)
    parser.add_argument("--hidden", default=64, type=int)
    parser.add_argument("--batch_size", default=8192, type=int)
    parser.add_argument("--dropout", default=0.05, type=float)
    parser.add_argument("--num_neg", default=3, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--interval", default=100, type=int)
    parser.add_argument("--step_lr_decay", action='store_true', default=True)
    parser.add_argument("--metric", default='hits@100', type=str)
    parser.add_argument("--gpu", default=0, type=int)
    parser.add_argument("--relu", action='store_true', default=False)
    parser.add_argument("--seed", default=0, type=int)
    parser.add_argument("--model", default='LightGCN', choices=['GCN', 'GCN_with_MLP', 'GCN_no_para', "LightGCN", "PureGCN"], type=str)
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
    parser.add_argument("--alpha", default=0.5, type=float)
    parser.add_argument("--use_val_as_input", action='store_true', default=False)
    parser.add_argument("--use_node_embedding", action='store_true', default=False)
    parser.add_argument("--only_node_embedding", action='store_true', default=False)
    parser.add_argument("--exp", action='store_true', default=False)
    parser.add_argument("--scale", action='store_true', default=False)
    parser.add_argument("--linear", action='store_true', default=False)
    parser.add_argument("--optimizer", default='adam', type=str)
    parser.add_argument("--activation", default='relu', type=str)  # New activation parameter.
    parser.add_argument("--init", default='orthogonal', type=str)
    parser.add_argument("--gin_aggr", default='mean', type=str)
    parser.add_argument("--multilayer", action='store_true', default=False)
    # New flag: if specified, run a wandb sweep.
    parser.add_argument("--sweep", action="store_true", help="Run hyperparameter sweep using wandb")
    args = parser.parse_args()
    return args

def eval_hits(y_pred_pos, y_pred_neg, K):
    if len(y_pred_neg) < K:
        return {'hits@{}'.format(K): 1.}
    kth_score_in_negative_edges = torch.topk(y_pred_neg, K)[0][-1]
    hitsK = float(torch.sum(y_pred_pos > kth_score_in_negative_edges).cpu()) / len(y_pred_pos)
    return {'hits@{}'.format(K): hitsK}

def eval_mrr(y_pred_pos, y_pred_neg):
    y_pred_pos = y_pred_pos.view(-1, 1)
    optimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    pessimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    ranking_list = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    mrr_list = 1. / ranking_list.to(torch.float)
    return {'mrr': mrr_list.mean().item(), 'mrr_list': mrr_list}

def adjustlr(optimizer, decay_ratio, lr):
    lr_ = lr * max(1 - decay_ratio, 0.0001)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr_

def train(model, g, train_pos_edge, optimizer, neg_sampler, pred, embedding=None):
    st = time.time()
    model.train()
    pred.train()
    dataloader = DataLoader(range(train_pos_edge.size(0)), args.batch_size, shuffle=True)
    total_loss = 0
    if args.maskinput:
        mask = torch.ones(train_pos_edge.size(0), dtype=torch.bool)
    for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
        if args.only_node_embedding:
            xemb = embedding.weight
        else:
            xemb = torch.cat((g.ndata['feat'], embedding.weight), dim=1) if args.use_node_embedding else g.ndata['feat']
        if args.maskinput:
            mask[edge_index] = 0
            tei = train_pos_edge[mask]
            src, dst = tei.t()
            re_tei = torch.stack((dst, src), dim=0).t()
            tei = torch.cat((tei, re_tei), dim=0)
            g_mask = dgl.graph((tei[:, 0], tei[:, 1]), num_nodes=g.num_nodes())
            g_mask = dgl.add_self_loop(g_mask)
            h = model(g_mask, xemb)
            mask[edge_index] = 1
        else:
            h = model(g, xemb)

        pos_edge = train_pos_edge[edge_index]
        neg_train_edge = neg_sampler(g, pos_edge.t()[0])
        neg_train_edge = torch.stack(neg_train_edge, dim=0).t()
        pos_score = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
        neg_score = pred(h[neg_train_edge[:, 0]], h[neg_train_edge[:, 1]])
        if args.loss == 'auc':
            loss = auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'hauc':
            loss = hinge_auc_loss(pos_score, neg_score, args.num_neg)
        elif args.loss == 'rank':
            loss = log_rank_loss(pos_score, neg_score, args.num_neg)
        else:
            loss = (F.binary_cross_entropy_with_logits(pos_score, torch.ones_like(pos_score)) +
                    F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score)))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(pred.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch time: {time.time() - st:.4f}", flush=True)
    return total_loss / len(dataloader)

def test(model, g, pos_test_edge, neg_test_edge, pred, embedding=None):
    model.eval()
    pred.eval()
    with torch.no_grad():
        if args.only_node_embedding:
            xemb = embedding.weight
        else:
            xemb = torch.cat((g.ndata['feat'], embedding.weight), dim=1) if args.use_node_embedding else g.ndata['feat']
        h = model(g, xemb)
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
        for k in [1, 3, 10, 100]:
            results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
        mrr_dict = eval_mrr(pos_score, neg_score.repeat(pos_score.size(0), 1))
        results['mrr'] = mrr_dict['mrr']
    return results

def eval_model(model, g, pos_valid_edge, neg_valid_edge, pos_train_edge, pred, embedding=None):
    model.eval()
    pred.eval()
    with torch.no_grad():
        if args.only_node_embedding:
            xemb = embedding.weight
        else:
            xemb = torch.cat((g.ndata['feat'], embedding.weight), dim=1) if args.use_node_embedding else g.ndata['feat']
        h = model(g, xemb)
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
        for k in [1, 3, 10, 100]:
            valid_results[f'hits@{k}'] = eval_hits(pos_score, neg_score, k)[f'hits@{k}']
        valid_results['mrr'] = eval_mrr(pos_score, neg_score.repeat(pos_score.size(0), 1))['mrr']
        dataloader = DataLoader(range(pos_train_edge.size(0)), args.batch_size)
        pos_score_train = []
        for _, edge_index in enumerate(tqdm.tqdm(dataloader)):
            pos_edge = pos_train_edge[edge_index]
            pos_pred = pred(h[pos_edge[:, 0]], h[pos_edge[:, 1]])
            pos_score_train.append(pos_pred)
        pos_score_train = torch.cat(pos_score_train, dim=0)
        train_results = {}
        for k in [1, 3, 10, 100]:
            train_results[f'hits@{k}'] = eval_hits(pos_score_train, neg_score, k)[f'hits@{k}']
        train_results['mrr'] = eval_mrr(pos_score_train, neg_score.repeat(pos_score_train.size(0), 1))['mrr']
    return train_results, valid_results

def train_test_split_edges(
    data,
    val_ratio: float = 0.1,
    test_ratio: float = 0.2,
):
    num_nodes = data.num_nodes()
    row, col = data.edges()

    # Return upper triangular portion.
    mask = row < col
    row, col = row[mask], col[mask]

    n_v = int(math.floor(val_ratio * row.size(0)))
    n_t = int(math.floor(test_ratio * row.size(0)))

    # Positive edges.
    perm = torch.randperm(row.size(0))
    row, col = row[perm], col[perm]

    r, c = row[:n_v], col[:n_v]
    data.val_pos_edge_index = torch.stack([r, c], dim=0)
    r, c = row[n_v:n_v + n_t], col[n_v:n_v + n_t]
    data.test_pos_edge_index = torch.stack([r, c], dim=0)

    r, c = row[n_v + n_t:], col[n_v + n_t:]
    data.train_pos_edge_index = torch.stack([r, c], dim=0)

    # Negative edges.
    neg_adj_mask = torch.ones(num_nodes, num_nodes, dtype=torch.uint8)
    neg_adj_mask = neg_adj_mask.triu(diagonal=1).to(torch.bool)
    neg_adj_mask[row, col] = 0

    neg_row, neg_col = neg_adj_mask.nonzero(as_tuple=False).t()
    perm = torch.randperm(neg_row.size(0))[:n_v + n_t]
    neg_row, neg_col = neg_row[perm], neg_col[perm]

    neg_adj_mask[neg_row, neg_col] = 0
    data.train_neg_adj_mask = neg_adj_mask

    row, col = neg_row[:n_v], neg_col[:n_v]
    data.val_neg_edge_index = torch.stack([row, col], dim=0)

    row, col = neg_row[n_v:n_v + n_t], neg_col[n_v:n_v + n_t]
    data.test_neg_edge_index = torch.stack([row, col], dim=0)

    return data

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
    if dataset == 'Cora':
        dataset = dgl.data.CoraGraphDataset()
    elif dataset == 'CiteSeer':
        dataset = dgl.data.CiteseerGraphDataset()
    elif dataset == 'PubMed':
        dataset = dgl.data.PubmedGraphDataset()
    data = dataset[0]
    split_edge = random_split_edges(data)
    graph = dgl.graph((split_edge['train']['edge'][:, 0], split_edge['train']['edge'][:, 1]), num_nodes=data.num_nodes())
    graph = dgl.to_bidirected(graph)
    graph = dgl.add_self_loop(graph)
    graph.ndata['feat'] = data.ndata['feat']
    return graph, split_edge

# =========================
# Experiment Function
# =========================
def run_experiment(run_idx, sweep_mode=False):
    """
    Runs one experiment (i.e. one training run).
    We now always call wandb.init() so that wandb.config is available.
    """
    # Set seeds.
    torch.manual_seed(args.seed + run_idx)
    torch.cuda.manual_seed_all(args.seed + run_idx)
    dgl.seed(args.seed + run_idx)

    # Always initialize wandb (with reinit=True) so that wandb.config is available.
    if sweep_mode:
        wandb.init(project='Refined-GAE', reinit=True, name=f'run_{run_idx}')
    else:
        wandb.init(project='Refined-GAE', config=args, reinit=True, name=f'run_{run_idx}')

    # Override defaults with wandb.config values (if present).

    if sweep_mode:
        config = wandb.config
        for key in config.keys():
            setattr(args, key, config[key])

    device = torch.device('cuda', args.gpu) if torch.cuda.is_available() else torch.device('cpu')

    graph, split_edge = load_data(args.dataset)
    graph = graph.to(device)
    train_pos_edge = split_edge['train']['edge'].to(device)
    valid_pos_edge = split_edge['valid']['edge'].to(device)
    valid_neg_edge = split_edge['valid']['edge_neg'].to(device)
    test_pos_edge = split_edge['test']['edge'].to(device)
    test_neg_edge = split_edge['test']['edge_neg'].to(device)

    if args.use_node_embedding or args.only_node_embedding:
        embedding = torch.nn.Embedding(graph.num_nodes(), args.emb_hidden).to(device)
        if args.init == 'orthogonal':
            torch.nn.init.orthogonal_(embedding.weight)
        elif args.init == 'ones':
            torch.nn.init.ones_(embedding.weight)
        elif args.init == 'random':
            torch.nn.init.uniform_(embedding.weight)
    else:
        embedding = None

    neg_sampler = GlobalUniform(args.num_neg)

    # --- Predictor ---
    if args.pred == 'dot':
        pred = DotPredictor().to(device)
    elif args.pred == 'mlp':
        pred = Hadamard_MLPPredictor(args.hidden, args.dropout, args.mlp_layers, args.res, args.norm, args.scale, args.activation).to(device)
    else:
        raise NotImplementedError

    if args.only_node_embedding:
        input_dim = args.emb_hidden
    else:
        input_dim = graph.ndata['feat'].shape[1] + args.emb_hidden if args.use_node_embedding else graph.ndata['feat'].shape[1]

    # --- Model ---
    if args.model == 'GCN':
        model = GCN_v1(input_dim, args.hidden, args.norm, args.relu, args.prop_step, args.dropout, 
                       args.multilayer, args.conv, args.res, args.gin_aggr).to(device)
    elif args.model == 'PureGCN':
        model = PureGCN(input_dim, args.prop_step, args.hidden, args.dropout, args.relu, args.norm, args.res).to(device)
    elif args.model == 'LightGCN':
        model = LightGCN(input_dim, args.hidden, args.prop_step, args.dropout, args.alpha, args.exp,
                         args.relu, args.norm, args.conv).to(device)
    else:
        raise NotImplementedError

    parameters = itertools.chain(model.parameters(), pred.parameters())
    if args.use_node_embedding:
        parameters = itertools.chain(parameters, embedding.parameters())

    # --- Optimizer ---
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(parameters, lr=args.lr)
    elif args.optimizer == 'adamw':
        optimizer = torch.optim.AdamW(parameters, lr=args.lr)
    else:
        raise NotImplementedError

    best_val = 0
    final_test_result = None
    best_epoch = 0
    losses = []
    valid_list = []
    test_list = []

    total_params = (sum(p.numel() for p in model.parameters()) +
                    sum(p.numel() for p in pred.parameters()))
    if args.use_node_embedding:
        total_params += sum(p.numel() for p in embedding.parameters())
    print(f'Run {run_idx+1}: number of parameters: {total_params}')

    # --- Training Loop ---
    for epoch in range(args.epochs):
        loss = train(model, graph, train_pos_edge, optimizer, neg_sampler, pred, embedding)
        losses.append(loss)
        if epoch % args.interval == 0 and args.step_lr_decay:
            adjustlr(optimizer, epoch / args.epochs, args.lr)
        train_results, valid_results = eval_model(model, graph, valid_pos_edge, valid_neg_edge, train_pos_edge, pred, embedding)
        valid_list.append(valid_results[args.metric])
        test_results = test(model, graph, test_pos_edge, test_neg_edge, pred, embedding)
        test_list.append(test_results[args.metric])
        if valid_results[args.metric] > best_val:
            best_val = valid_results[args.metric]
            best_epoch = epoch
            final_test_result = test_results
        if epoch - best_epoch >= 200:
            break
        print(f"Epoch {epoch}, Loss: {loss:.4f}, Train {args.metric}: {train_results[args.metric]:.4f}, "
              f"Valid {args.metric}: {valid_results[args.metric]:.4f}, Test {args.metric}: {test_results[args.metric]:.4f}")
        wandb.log({'train_' + args.metric: train_results[args.metric],
                   'valid_' + args.metric: valid_results[args.metric],
                   'test_' + args.metric: test_results[args.metric],
                   'loss': loss})
    print("Final Test Results for this run:")
    for k, v in final_test_result.items():
        print(f"{k}: {v:.4f}", end=' ')
    print(f"\nTest {args.metric}: {final_test_result[args.metric]:.4f}")

    wandb.log({'final_' + args.metric: final_test_result[args.metric]})
    wandb.finish()
    return final_test_result

# =========================
# Standard run (averaging 5 runs)
# =========================
def run_once():
    num_runs = 5
    final_results_list = []
    for run_idx in range(num_runs):
        print(f"\nStarting run {run_idx+1}/{num_runs}")
        result = run_experiment(run_idx, sweep_mode=False)
        final_results_list.append(result)
    avg_results = {}
    stds = {}
    keys = final_results_list[0].keys()
    for key in keys:
        avg_results[key] = sum(result[key] for result in final_results_list) / num_runs
        stds[key] = math.sqrt(sum((result[key] - avg_results[key])**2 for result in final_results_list) / num_runs)
    print("\nAverage final test results over 5 runs:")
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}", end=' ')
    for key, value in stds.items():
        print(f"{key}_std: {value:.4f}", end=' ')
    print()
    return avg_results

# =========================
# Sweep Agent Functions
# =========================
def sweep_train():
    """
    Called by the wandb agent. In sweep mode we run 3 experiments (to average randomness)
    and log the averaged final metrics.
    """
    num_runs = 5
    final_results_list = []
    for run_idx in range(num_runs):
        print(f"\nSweep mode: starting sub-run {run_idx+1}/{num_runs}")
        result = run_experiment(run_idx, sweep_mode=True)
        final_results_list.append(result)
    avg_results = {}
    keys = final_results_list[0].keys()
    for key in keys:
        avg_results[key] = sum(result[key] for result in final_results_list) / num_runs
    print("\nSweep mode: Average final test results over 5 runs:")
    for key, value in avg_results.items():
        print(f"{key}: {value:.4f}", end=' ')
    print()
    temp_run = wandb.init(project='Refined-GAE', reinit=True)
    wandb.log({'avg_' + args.metric: avg_results[args.metric]})
    wandb.finish()

def main_sweep():
    """
    Set up and run a wandb hyperparameter sweep.
    """
    global args
    args = parse()
    # Define the sweep configuration with your updated hyperparameter values.
    sweep_config = {
        'method': 'bayes',  # can also be "grid" or "random"
        'metric': {
            'name': 'avg_' + args.metric,  # e.g., final_hits@20
            'goal': 'maximize'
        },
        'parameters': {
            'lr': {'values': [0.001, 0.005, 0.01]},
            'hidden': {'values': [256, 512, 1024]},
            'dropout': {'values': [0.2, 0.6]},
            'prop_step': {'values': [2, 4]},
            'batch_size': {'values': [2048, 4096, 8192]},
            'norm': {'values': [False, True]},
            'optimizer': {'values': ['adam', 'adamw']},
            'activation': {'values': ['relu', 'gelu', 'silu']},
            'maskinput': {'values': [False, True]},
            'mlp_layers': {'values': [2, 4, 6]},
            'use_node_embedding': {'values': [True]},
            # Additional hyperparameters can be added here.
        }
    }
    # Create a new sweep.
    sweep_id = wandb.sweep(sweep_config, project='Refined-GAE')
    # Launch the sweep agent.
    wandb.agent(sweep_id, function=sweep_train)

def main():
    global args
    args = parse()
    print(args)
    # If the sweep flag is set, run the sweep; otherwise run the standard experiment.
    if args.sweep:
        main_sweep()
        return
    run_once()

if __name__ == "__main__":
    main()
