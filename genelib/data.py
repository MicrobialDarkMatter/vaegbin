from glob import glob
#from natsort import natsorted
import numpy as np
import os
import scipy.sparse

def load_reference_markers(path):
    return eval(' '.join(open(path).readlines()).split('\t')[-1].strip())

def load_contig_genes(path):
    """Open file mapping contigs to genes

    :param contig_markers: path to contig markers (marker stats)
    :type contig_markers: str
    :return: Mapping contig names to markers
    :rtype: dict
    """
    contigs = {}
    for line in open(path, "r").readlines():
        values = line.strip().split("\t")
        contig_name = values[0]
        # keep only first two elements
        contig_name = "_".join(contig_name.split("_")[:2])
        contigs[contig_name] = {}
        mappings = eval(values[1])
        for contig in mappings:
            for gene in mappings[contig]:
                if gene not in contigs[contig_name]:
                    contigs[contig_name][gene] = 0
                contigs[contig_name][gene] += 1
                if len(mappings[contig][gene]) > 1:
                    print("contig with multiple copies of gene:", contig, gene)
                    pass
    return contigs

def load_dataset(path):
    prefix = os.path.join(path, "{}")
    adj_sparse = scipy.sparse.load_npz(prefix.format('adj_sparse.npz'))
    edge_weights = np.load(prefix.format('edge_weights.npy'))
    node_features = []
    
    for fname in glob(prefix.format('node_attributes*.npy')):
        F = np.load(fname)
        if len(F.shape) == 1:
            F = F[:,None]
        if len(node_features):
            assert len(F) == len(node_features[-1])
        node_features.append(F)
    node_features = np.hstack(node_features)
    node_names = np.load(prefix.format('node_names.npy'))
    return node_features, node_names, adj_sparse, edge_weights


