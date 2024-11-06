This repository contains the code for the paper "Reconsidering the performance of GAE in link prediction" by Weishuo Ma, Yanbo Wang, XiYuan Wang and Muhan Zhang.

The code is based on the DGL library and the OGB library. To run the code, you need to set up the environment specified in the env.yaml file.

Below we give the commands to run the code on the four datasets in the OGB benchmark.


```python train_wo_feat.py --dataset ogbl-ddi --lr 0.001 --hidden 1024 --batch_size 8192 --dropout 0.6 --num_neg 1 --epochs 500 --prop_step 2 --metric hits@20 --residual 0.1 --maskinput --mlp_layers 8 --mlp_res```

```python train_w_feat.py --dataset ogbl-collab --lr 0.0004 --emb_hidden 512 --gnn_hidden 1024 --mlp_hidden 1024--batch_size 16384 --dropout 0.6 --num_neg 6 --epochs 800 --prop_step 4 --metric hits@50 --mlp_layers 5```

```python train_wo_feat.py --dataset ogbl-ppa --lr 0.001 --hidden 512 --batch_size 65536 --dropout 0.2 --num_neg 6 --epochs 500 --prop_step 2 --metric hits@100 --residual 0.1```

```python train_w_feat.py --dataset ogbl-citation2 --lr 0.003 --emb_hidden 128 --gnn_hidden 128 --mlp_hidden 128 --batch_size 131072 --dropout 0 --num_neg 6 --epochs 200 --prop_step 3 --metric MRR```


For ogbl-citation2 dataset, you need a GPU with at least 40GB memory.