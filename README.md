# VAEG-BIN: Graph Neural Networks for Metagenomic Binning

# Data

Each dataset should be in ./data and have the following files:

- node_names.npy  
- adj_sparse.npz 
- edge_weights.npy  
- node_features.npy  

If simulated dataset:

- node_to_label.npy
- label_to_node.npy  
- labels.npy

If real dataset:

- marker_gene_stats.tsv

The original data can be found at https://zenodo.org/record/6122610


# Run command

```
python train.py <dataset_name>
```

# Note about Results

With the simulated Strong100 dataset, the values calculated during training do not correspond to the final evaluation.
For that, use the [AMBER tool](https://github.com/CAMI-challenge/AMBER) on the tsv file generated after training with the *gold_standard_genome.tsv* file. It will generate a file called index.html with the actual AP, AR and F1 values as in the paper. 
For example:
```
amber.py  -l "GCN SAGE GAT" -g /host/gold_standard_genome.tsv -o /host/results/ /host/strong100_GCN_vamb0_1l_DIFF-C_results.tsv /host/strong100_SAGE_vamb0_1l_DIFF-C_results.tsv /host/strong100_GAT_vamb0_1l_DIFF-C_results.tsv
```
