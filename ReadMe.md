**Update: We are thrilled to announce that our paper, "Reconsidering the Performance of GAE in Link Prediction," has been selected for the 🏆 Best Full Paper Award 🏆 at the 34th ACM International Conference on Information and Knowledge Management (CIKM 2025)!**

You can read the pre-print on arXiv:

https://arxiv.org/abs/2411.03845

We achieve comparable or better performance than recent models on the OGB benchmark datasets, including ogbl-ddi, ogbl-collab, ogbl-ppa, and ogbl-citation2:

| Metric        | Cora          | Citeseer      | Pubmed        | Collab        | PPA           | Citation2     | DDI           |
|---------------|---------------|---------------|---------------|---------------|---------------|---------------|---------------|
|               | Hits@100      | Hits@100      | Hits@100      | Hits@50       | Hits@100      | MRR           | Hits@20       |
| **CN** | $33.92 \pm 0.46$ | $29.79 \pm 0.90$ | $23.13 \pm 0.15$ | $56.44 \pm 0.00$ | $27.65 \pm 0.00$ | $51.47 \pm 0.00$ | $17.73 \pm 0.00$ |
| **AA** | $39.85 \pm 1.34$ | $35.19 \pm 1.33$ | $27.38 \pm 0.11$ | $64.35 \pm 0.00$ | $32.45 \pm 0.00$ | $51.89 \pm 0.00$ | $18.61 \pm 0.00$ |
| **RA** | $41.07 \pm 0.48$ | $33.56 \pm 0.17$ | $27.03 \pm 0.35$ | $64.00 \pm 0.00$ | $49.33 \pm 0.00$ | $51.98 \pm 0.00$ | $27.60 \pm 0.00$ |
| **SEAL** | $81.71 \pm 1.30$ | $83.89 \pm 2.15$ | $75.54 \pm 1.32$ | $64.74 \pm 0.43$ | $48.80 \pm 3.16$ | $87.67 \pm 0.32$ | $30.56 \pm 3.86$ |
| **NBFNet** | $71.65 \pm 2.27$ | $74.07 \pm 1.75$ | $58.73 \pm 1.99$ | OOM           | OOM           | OOM           | $4.00 \pm 0.58$  |
| **Neo-GNN** | $80.42 \pm 1.31$ | $84.67 \pm 2.16$ | $73.93 \pm 1.19$ | $57.52 \pm 0.37$ | $49.13 \pm 0.60$ | $87.26 \pm 0.84$ | $63.57 \pm 3.52$ |
| **BUDDY** | $88.00 \pm 0.44$ | $\mathbf{92.93 \pm 0.27}$ | $74.10 \pm 0.78$ | $65.94 \pm 0.58$ | $49.85 \pm 0.20$ | $87.56 \pm 0.11$ | $78.51 \pm 1.36$ |
| **NCN** | $\mathbf{89.05 \pm 0.96}$ | $91.56 \pm 1.43$ | $\underline{79.05 \pm 1.16}$ | $64.76 \pm 0.87$ | $61.19 \pm 0.85$ | $88.09 \pm 0.06$ | $\underline{82.32 \pm 6.10}$ |
| **MPLP+** | -             | -             | -             | $\mathbf{66.99 \pm 0.40}$ | $\underline{65.24 \pm 1.50}$ | $\mathbf{90.72 \pm 0.12}$ | -             |
| **GAE(GCN)** | $66.79 \pm 1.65$ | $67.08 \pm 2.94$ | $53.02 \pm 1.39$ | $47.14 \pm 1.45$ | $18.67 \pm 1.32$ | $84.74 \pm 0.21$ | $37.07 \pm 5.07$ |
| **GAE(SAGE)** | $55.02 \pm 4.03$ | $57.01 \pm 3.74$ | $39.66 \pm 0.72$ | $54.63 \pm 1.12$ | $16.55 \pm 2.40$ | $82.60 \pm 0.36$ | $53.90 \pm 4.74$ |
| **Optimized-GAE**| $\underline{88.17 \pm 0.93}$ | $\underline{92.40 \pm 1.23}$ | $\mathbf{80.09 \pm 1.72}$ | $\underline{66.11 \pm 0.35}$ | $\mathbf{78.41 \pm 0.83}$ | $\underline{88.74 \pm 0.06}$ | $\mathbf{94.43 \pm 0.57}$ |

The code is based on the DGL library and the OGB library. To run the code, you need to set up the environment specified in the env.yaml file:

```conda env create -f env.yaml```

```python train_w_feat_small.py --dataset Cora --activation silu --batch_size 2048 --dropout 0.6 --hidden 1024 --lr 0.005 --maskinput --mlp_layers 4 --res --norm --num_neg 3 --optimizer adamw --prop_step 4 --model LightGCN```

```python train_w_feat_small.py --dataset CiteSeer --activation relu --batch_size 4096 --dropout 0.6 --hidden 1024 --lr 0.001 --maskinput --norm --prop_step 4 --num_neg 1```

```train_w_feat_small.py --dataset PubMed --activation gelu --batch_size 4096 --dropout 0.4 --exp --hidden 512 --lr 0.001 --maskinput --mlp_layers 2 --norm --num_neg 3 --prop_step 2 --model LightGCN```

Below we give the commands to run the code on the four datasets in the OGB benchmark.

```python train_wo_feat.py --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 --dropout 0.6 --num_neg 1 --epochs 500 --prop_step 2 --metric hits@20 --residual 0.1 --maskinput --mlp_layers 8 --mlp_res --emb_dim 1024```

```python collab.py --dataset ogbl-collab --lr 0.0004 --emb_hidden 0 --hidden 1024 --batch_size 16384 --dropout 0.2 --num_neg 3 --epoch 500 --prop_step 4 --metric hits@50 --mlp_layers 5 --res --norm --dp4norm 0.2 --scale```

```python train_wo_feat.py --dataset ogbl-ppa --lr 0.001 --hidden 512 --batch_size 65536 --dropout 0.2 --num_neg 3 --epoch 800 --prop_step 2 --metric hits@100 --residual 0.1 --mlp_layers 5 --mlp_res --emb_dim 512```

```python citation.py --dataset ogbl-citation2 --lr 0.0003 --clip_norm 1 --emb_hidden 256 --hidden 256 --batch_size 65536 --dropout 0.2 --num_neg 3 --epochs 200 --prop_step 3 --metric MRR --norm --dp4norm 0.2 --mlp_layers 5```

For ogbl-citation2 dataset, you need a GPU with at least 40GB memory.

