# Cooperating local and global feature to enhance the expressiveness of GNN
This is the code for the project of Graph Nerual Network in PKU. 
The code is implemented based on [**GNN-AK**](https://github.com/LingxiaoShawn/GNNAsKernel).

## Setup 

```
# params
# 10/6/2022, newest packages. 
ENV=gnn_ak

# create env 
conda create --name $ENV python=3.10 -y
conda activate $ENV

# install pytorch 
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch

# install pyg (version 2.1.0)
conda install pyg -c pyg

# install ogb 
pip install ogb

# install rdkit
conda install -c conda-forge rdkit -y

# update yacs and tensorboard
pip install yacs==0.1.8 --force  # PyG currently use 0.1.6 which doesn't support None argument. 
pip install tensorboard
pip install matplotlib

# install networkx and DGL
pip install networkx==2.6.3
pip install dgl==0.2

```

## Code structure
``core/`` contains all source code.   
``train/`` contains all scripts for available datasets.  

* Subgraph extraction is implemented as data transform operator in PyG. See ``core/transform.py``. The transform layer will built the mapping from original nodes and edges to all subgraphs.  
* The main part of our method is implemented in ``core/model.py``. We use networkx and DGL to implemente global topology feature extraction.

## Dataset
We use [**ZINC**](https://pubs.acs.org/doi/full/10.1021/acs.jcim.5b00559) dataset.

## Hyperparameters 

See ``core/config.py`` for all options. 


## Run with different subgraph sampling hops

```
# Run with different subgraph sampling hops
python -m train.zinc model.mini_layers 1 subgraph.hops 1 model.gnn_type GCNConv
python -m train.zinc model.mini_layers 1 subgraph.hops 2 model.gnn_type GCNConv
python -m train.zinc model.mini_layers 1 subgraph.hops 3 model.gnn_type GCNConv
python -m train.zinc model.mini_layers 1 subgraph.hops 4 model.gnn_type GCNConv
python -m train.zinc model.mini_layers 1 subgraph.hops 5 model.gnn_type GCNConv
```
## Add global topology feature

```
# Use PageRank value as global topology feature
python -m train.zinc model.mini_layers 1 subgraph.hops 3 model.use_page_rank True model.gnn_type GCNConv

# Use Distance Encoding as global topology feature (MEAN and MAX)
python -m train.zinc model.mini_layers 1 subgraph.hops 3 model.use_distance_encoding True model.use_de_mean True model.gnn_type GCNConv
python -m train.zinc model.mini_layers 1 subgraph.hops 3 model.use_distance_encoding True model.use_de_max True model.gnn_type GCNConv
```



<!--
**GNNAsKernel/GNNAsKernel** is a ✨ _special_ ✨ repository because its `README.md` (this file) appears on your GitHub profile.

Here are some ideas to get you started:

- 🔭 I’m currently working on ...
- 🌱 I’m currently learning ...
- 👯 I’m looking to collaborate on ...
- 🤔 I’m looking for help with ...
- 💬 Ask me about ...
- 📫 How to reach me: ...
- 😄 Pronouns: ...
- ⚡ Fun fact: ...
-->
