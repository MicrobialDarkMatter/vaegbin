from sklearn.cluster import KMeans
import sys
import os
from tqdm import tqdm
import itertools
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from rich.console import Console
from rich.table import Table
from scipy.sparse import csr_matrix, diags

console = Console()
print = Console().log

from genelib.data import load_reference_markers, load_contig_genes, load_dataset
from genelib.scores import compute_hq, calculate_overall_prf
from genelib.models import SAGE, GCN, GAT, TH
from genelib.vamb_clustering import cluster as vamb_cluster

DO_PLAIN_FEATURE_CLUSTERING = False

RESULT_EVERY = 50


def get_all_different_idx(node_names, contig_genes):
    """
    Returns a 2d numpy array where each row
    corresponds to a pairs of node idx whose
    feature must be different as they correspond
    to the same contig (check jargon). This
    should encourage the HQ value to be higher.
    """
    node_names_to_idx = {node_name: i for i, node_name in enumerate(node_names)}
    pair_idx = set()
    for n1 in contig_genes:
        for gene1 in contig_genes[n1]:
            for n2 in contig_genes:
                if n1 != n2 and gene1 in contig_genes[n2]:
                    p1 = (node_names_to_idx[n1], node_names_to_idx[n2])
                    p2 = (node_names_to_idx[n2], node_names_to_idx[n1])
                    if (p1 not in pair_idx) and (p2 not in pair_idx):
                        pair_idx.add(p1)
    pair_idx = np.unique(np.array(list(pair_idx)), axis=0)
    print("Number of diff cluster pairs:", len(pair_idx))
    return pair_idx


def normalize_adj(A):
    A[A > 0] = 1
    A[np.diag_indices(len(A))] = 1
    rowsum = A.sum(axis=1)
    D = np.diag(np.power(rowsum, -0.5))
    A_norm = A.dot(D).T.dot(D)
    A_norm = A_norm.astype(np.float32)
    return A_norm


def normalize_adj_sparse(A):
    # https://github.com/tkipf/gcn/blob/master/gcn/utils.py
    A.setdiag(1)
    rowsum = np.array(A.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = diags(d_inv_sqrt)
    return A.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()


def compute_clusters_and_stats(
    X, node_names, reference_markers, contig_genes, node_to_gt_idx_label, gt_idx_label_to_node, k=0, clustering="vamb"
):
    if clustering == "vamb":
        physical_devices = tf.config.list_physical_devices("GPU")
        use_cuda = len(physical_devices) > 0
        best_cluster_to_contig = {i: c for (i, (n, c)) in enumerate(vamb_cluster(X.astype(np.float32), node_names))}

        best_contig_to_bin = {}
        for b in best_cluster_to_contig:
            for contig in best_cluster_to_contig[b]:
                best_contig_to_bin[contig] = b
        labels = np.array([best_contig_to_bin[n] for n in node_names])
    elif clustering == "kmeans":
        clf = KMeans(k, random_state=1234)
        labels = clf.fit_predict(X)
        best_contig_to_bin = {node_names[i]: labels[i] for i in range(len(node_names))}
        best_cluster_to_contig = {i: [] for i in range(k)}
        for i in range(len(node_names)):
            best_cluster_to_contig[labels[i]].append(node_names[i])
    hq, positive_clusters = compute_hq(
        reference_markers=reference_markers, contig_genes=contig_genes, node_names=node_names, node_labels=labels
    )
    mq, _ = compute_hq(
        reference_markers=reference_markers,
        contig_genes=contig_genes,
        node_names=node_names,
        node_labels=labels,
        comp_th=50,
        cont_th=10,
    )
    non_comp, _ = compute_hq(
        reference_markers=reference_markers,
        contig_genes=contig_genes,
        node_names=node_names,
        node_labels=labels,
        comp_th=0,
        cont_th=10,
    )
    all_cont, _ = compute_hq(
        reference_markers=reference_markers,
        contig_genes=contig_genes,
        node_names=node_names,
        node_labels=labels,
        comp_th=90,
        cont_th=1000,
    )
    # print(hq, mq, "incompete but non cont:", non_comp, "cont but complete:", all_cont)
    positive_pairs = []
    node_names_to_idx = {node_name: i for i, node_name in enumerate(node_names)}
    for label in positive_clusters:
        for (p1, p2) in itertools.combinations(best_cluster_to_contig[label], 2):
            positive_pairs.append((node_names_to_idx[p1], node_names_to_idx[p2]))
    # print("found {} positive pairs".format(len(positive_pairs)))
    positive_pairs = np.unique(np.array(list(positive_pairs)), axis=0)
    if node_to_gt_idx_label is not None:
        p, r, f1, ari = calculate_overall_prf(
            best_cluster_to_contig, best_contig_to_bin, node_to_gt_idx_label, gt_idx_label_to_node
        )
    else:
        p, r, f1, ari = 0, 0, 0, 0

    return (
        labels,
        {"precision": p, "recall": r, "f1": f1, "ari": ari, "hq": hq, "mq": mq, "n_clusters": len(np.unique(labels))},
        positive_pairs,
        positive_clusters,
    )


class ResultTable:
    def __init__(self, n_real_labels):
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("Model", style="dim", width=12)
        table.add_column("Precision")
        table.add_column("Recall", justify="left")
        table.add_column("F1", justify="left")
        table.add_column("ARI", justify="left")
        table.add_column(f"HQ", justify="left")
        table.add_column(f"MQ", justify="left")
        table.add_column(f"#Clusters (GT Labels=[red]{n_real_labels}[/red])", justify="right")
        self.table = table
        self.rows = []

    def add_row(self, model_name, S):
        df = pd.DataFrame(S)
        mean = df.mean()
        std = df.std()
        r = (
            [model_name]
            + [f"{mean[c]:.2f}±{std[c]:.2f}" for c in df.columns[:-3]]
            + [f"{int(mean[c]):d}±{int(std[c]):d}" for c in df.columns[-3:]]
        )
        print(r)
        self.table.add_row(*r, end_section=True)
        self.rows.append(",".join(r))

    def show(self):
        console.print(self.table)
        for r in self.rows:
            print(r)


def filter_graph_with_markers(adj, node_names, markers, edge_features, depth=2):
    adj.data = edge_features
    graph = nx.convert_matrix.from_scipy_sparse_matrix(adj, edge_attribute="weight")
    total_edges = len(graph.edges())
    same_marker_edges = []
    same_marker_weights = []
    paths_with_markers = 0
    edges_to_remove = set()
    visited_nodes = set()
    for n1 in graph.nodes:
        visited_nodes.add(n1)
        visited_nodes2 = set()
        if node_names[n1] in markers:
            for n2 in graph.neighbors(n1):
                visited_nodes2.add(n2)
                if node_names[n2] in markers:
                    paths_with_markers += 1
                    marker_intersection = len(markers[node_names[n1]].keys() & markers[node_names[n2]].keys())
                    if marker_intersection > 0:
                        same_marker_edges.append(marker_intersection)
                        same_marker_weights.append(graph.edges[n1, n2]["weight"])
                        edges_to_remove.add((n1, n2))
                # check second order
                for n3 in graph.neighbors(n2):
                    if node_names[n3] in markers and n3 not in visited_nodes:
                        paths_with_markers += 1
                        marker_intersection = len(markers[node_names[n1]].keys() & markers[node_names[n3]].keys())
                        if marker_intersection > 0:
                            same_marker_edges.append(marker_intersection)
                            edges_to_remove.add((n1, n2))
                            edges_to_remove.add((n2, n3))
                    if n3 not in visited_nodes and depth > 2:
                        for n4 in graph.neighbors(n3):
                            if node_names[n4] in markers and n4 not in visited_nodes2:
                                paths_with_markers += 1
                                marker_intersection = len(
                                    markers[node_names[n1]].keys() & markers[node_names[n4]].keys()
                                )
                                if marker_intersection > 0:
                                    same_marker_edges.append(marker_intersection)
                                    edges_to_remove.add((n1, n2))
                                    edges_to_remove.add((n2, n3))
                                    edges_to_remove.add((n2, n4))

    graph.remove_edges_from(list(edges_to_remove))
    adj = nx.to_scipy_sparse_matrix(graph, format="coo")
    print("old edges:", total_edges, "new edges:", len(graph.edges()), "removed", len(edges_to_remove), "depth", depth)
    return adj, adj.data


def filter_disconnected(adj, node_names, markers):
    # get idx of nodes that are connected or have at least one marker
    graph = nx.convert_matrix.from_scipy_sparse_matrix(adj, edge_attribute="weight")
    # breakpoint()
    nodes_to_remove = set()
    for n1 in graph.nodes:
        if len(list(graph.neighbors(n1))) == 0 and (
            node_names[n1] not in markers or len(markers[node_names[n1]]) == 0
        ):
            nodes_to_remove.add(n1)
    graph.remove_nodes_from(list(nodes_to_remove))
    print(len(nodes_to_remove), "out of", len(node_names), "nodes without edges and markers")
    return set(graph.nodes())


def plot_clusters(node_features, vae_features, node_labels, model_name):
    plt.figure()
    X_hat = node_features[np.argsort(node_labels)]
    vae_features_sorted = vae_features[np.argsort(node_labels)]
    gnn_dist_matrix = tf.sigmoid(np.dot(X_hat, X_hat.T))
    vae_dist_matrix = tf.sigmoid(np.dot(vae_features_sorted, vae_features_sorted.T))
    plt.imshow(gnn_dist_matrix + vae_dist_matrix)
    plt.colorbar()
    plt.savefig(f"plots/{model_name}.png", format="png", dpi=150)
    plt.close()


if __name__ == "__main__":
    n_runs = 10
    dataset = sys.argv[1]
    reference_markers = load_reference_markers("data/Bacteria.ms")
    contig_genes = load_contig_genes("data/{}/marker_gene_stats.tsv".format(dataset))
    node_raw, node_names, adjacency_matrix_sparse, edge_features = load_dataset("data/{}/".format(dataset))
    node_features = np.load("data/{}/node_features.npy".format(dataset))

    # normalize nodes
    node_raw = (node_raw - node_raw.mean(axis=0, keepdims=True)) / node_raw.std(axis=0, keepdims=True)
    node_features = (node_features - node_features.mean(axis=0, keepdims=True)) / node_features.std(
        axis=0, keepdims=True
    )

    depth = 2
    adjacency_matrix_sparse, edge_features = filter_graph_with_markers(
        adjacency_matrix_sparse, node_names, contig_genes, edge_features, depth=depth
    )

    connected_marker_nodes = filter_disconnected(adjacency_matrix_sparse, node_names, contig_genes)
    nodes_with_markers = [i for i, n in enumerate(node_names) if n in contig_genes and len(contig_genes[n]) > 0]
    # For calculating P, R, F1, ARI
    if os.path.exists("data/{}/labels.npy".format(dataset)):
        node_to_gt_idx_label = np.load("data/{}/node_to_label.npy".format(dataset), allow_pickle=True)[()]
        gt_idx_label_to_node = np.load("data/{}/label_to_node.npy".format(dataset), allow_pickle=True)[()]
        gt_labels = np.load("data/{}/labels.npy".format(dataset))
        res_table = ResultTable(len(gt_labels))

    else:
        node_to_gt_idx_label, gt_idx_label_to_node, gt_labels = None, None, None
        res_table = ResultTable(1)
    ####
    if os.path.exists("data/{}/all_different.npy".format(dataset)):
        all_different_idx = np.load("data/{}/all_different.npy".format(dataset))
    else:
        all_different_idx = get_all_different_idx(node_names, contig_genes)
        np.save("data/{}/all_different.npy".format(dataset), all_different_idx)

    if DO_PLAIN_FEATURE_CLUSTERING:
        S = []
        for i in range(2):
            cluster_labels, stats, all_same_idx, positive_clusters = compute_clusters_and_stats(
                node_raw, node_names, reference_markers, contig_genes, node_to_gt_idx_label, gt_idx_label_to_node
            )
            S.append(stats)
        res_table.add_row("VAMB RAW-F", S)

        # plot_clusters(node_raw, cluster_labels, "VAMB RAW-F")

        S = []
        for i in range(2):
            cluster_labels, stats, all_same_idx, positive_clusters = compute_clusters_and_stats(
                node_features, node_names, reference_markers, contig_genes, node_to_gt_idx_label, gt_idx_label_to_node
            )
            S.append(stats)
        res_table.add_row("VAMB VAE-F", S)
        # plot_clusters(node_features, cluster_labels, "VAMB VAE-F")

    adj_norm = normalize_adj_sparse(adjacency_matrix_sparse)

    hidden_units = 128
    output_dim = 64
    epochs = 500
    lr = 1e-2
    nlayers = 2
    VAE = False
    clustering = "vamb"
    k = 0
    use_edge_weights = True
    use_disconnected = True
    cluster_markers_only = False
    decay = 0.5 ** (2.0 / epochs)
    concat_features = True
    print("using edge weights", use_edge_weights)
    print("using disconnected", use_disconnected)
    print("concat features", concat_features)
    print("cluster markers only", cluster_markers_only)
    # tf.config.experimental_run_functions_eagerly(True)
    if cluster_markers_only:
        print("eval with ", len(nodes_with_markers), "contigds")
        cluster_mask = [n in nodes_with_markers for n in range(len(node_names))]
    else:
        cluster_mask = [True] * len(node_names)
    if use_edge_weights:
        edge_features = (edge_features - edge_features.min()) / (edge_features.max() - edge_features.min())
        old_rows, old_cols = adjacency_matrix_sparse.row, adjacency_matrix_sparse.col
        old_idx_to_edge_idx = {(r, c): i for i, (r, c) in enumerate(zip(old_rows, old_cols))}
        old_values = adj_norm.data.astype(np.float32)
        new_values = []
        for i, j, ov in zip(adj_norm.row, adj_norm.col, old_values):
            if i == j:
                new_values.append(1.0)
            else:
                try:
                    eidx = old_idx_to_edge_idx[(i, j)]
                    new_values.append(ov * edge_features[eidx])
                except:
                    new_values.append(ov)
        new_values = np.array(new_values).astype(np.float32)
    else:
        new_values = adj_norm.data.astype(np.float32)
    adj = tf.SparseTensor(
        indices=np.array([adj_norm.row, adj_norm.col]).T, values=new_values, dense_shape=adj_norm.shape
    )
    adj = tf.sparse.reorder(adj)
    if not use_disconnected:
        train_edges = [
            eid
            for eid in range(len(adj_norm.row))
            if adj_norm.row[eid] in connected_marker_nodes and adj_norm.col[eid] in connected_marker_nodes
        ]

        train_adj = tf.SparseTensor(
            indices=np.array([adj_norm.row[train_edges], adj_norm.col[train_edges]]).T,
            values=new_values[train_edges],
            dense_shape=adj_norm.shape,
        )
        train_adj = tf.sparse.reorder(train_adj)

    else:
        train_adj = adj
    print("train len edges:", train_adj.indices.shape[0])
    for nlayers in range(1, 4):
        X = node_features
        print("feat dim", X.shape)
        for pname, neg_pair_idx, pos_pair_idx in [("DIFF-C", all_different_idx, None)]:
            # for pname, neg_pair_idx, pos_pair_idx in [("DIFF-C", all_different_idx, None)]:
            for gname, gmodel in [
                ("GCN", GCN),
                ("SAGE", SAGE),
                ("GAT", GAT),
            ]:
                scores = []
                all_cluster_labels = []
                features = tf.constant(X.astype(np.float32))
                S = []
                for i in range(n_runs):
                    model = gmodel(
                        features=features,
                        labels=None,
                        adj=train_adj,
                        n_labels=output_dim,
                        hidden_units=hidden_units,
                        layers=nlayers,
                        conv_last=False,
                    )
                    th = TH(model, lr=lr, lambda_vae=0.01, all_different_idx=neg_pair_idx, all_same_idx=pos_pair_idx)
                    train_idx = np.arange(len(features))
                    pbar = tqdm(range(epochs))
                    scores = []
                    for e in pbar:
                        if VAE:
                            loss = th.train_unsupervised_vae(train_idx)
                        else:
                            loss, same_loss, diff_loss = th.train_unsupervised(train_idx)
                            # loss = th.train_unsupervised_v2(train_idx)
                        loss = loss.numpy()
                        pbar.set_description(f"[{i} {gname} {nlayers}l {pname}] L={loss:.3f}")
                        if (e + 1) % RESULT_EVERY == 0:
                            model.adj = adj
                            node_new_features = model(None, training=False)
                            node_new_features = node_new_features.numpy()
                            node_new_features = node_new_features[:, :output_dim]
                            # concat with original features
                            if concat_features:
                                node_new_features = tf.concat([features, node_new_features], axis=1).numpy()
                            cluster_labels, stats, _, _ = compute_clusters_and_stats(
                                node_new_features[cluster_mask],
                                node_names[cluster_mask],
                                reference_markers,
                                contig_genes,
                                node_to_gt_idx_label,
                                gt_idx_label_to_node,
                                clustering=clustering,
                                k=k,
                            )
                            # print(f'--- EPOCH {e:d} ---')
                            # print(stats)
                            scores.append(stats)
                            all_cluster_labels.append(cluster_labels)
                            model.adj = train_adj
                            # print('--- END ---')
                    model.adj = adj
                    node_new_features = model(None, training=False)
                    node_new_features = node_new_features.numpy()
                    node_new_features = node_new_features[:, :output_dim]
                    # concat with original features
                    if concat_features:
                        node_new_features = tf.concat([features, node_new_features], axis=1).numpy()

                    cluster_labels, stats, _, _ = compute_clusters_and_stats(
                        node_new_features[cluster_mask],
                        node_names[cluster_mask],
                        reference_markers,
                        contig_genes,
                        node_to_gt_idx_label,
                        gt_idx_label_to_node,
                        clustering=clustering,
                        k=k,
                    )
                    scores.append(stats)
                    # get best stats:
                    if concat_features:  # use HQ
                        hqs = [s["hq"] for s in scores]
                        best_idx = np.argmax(hqs)
                    else:  # use F1
                        f1s = [s["f1"] for s in scores]
                        best_idx = np.argmax(f1s)
                    # S.append(stats)
                    S.append(scores[best_idx])
                    # print(f"best epoch: {RESULT_EVERY + (best_idx*RESULT_EVERY)} : {scores[best_idx]}")
                    with open(f"{dataset}_{gname}_{clustering}{k}_{nlayers}l_{pname}_{i}_results.tsv", "w") as f:
                        f.write("@Version:0.9.0\n@SampleID:SAMPLEID\n@@SEQUENCEID\tBINID\n")
                        for i in range(len(cluster_labels)):
                            f.write(f"{node_names[i]}\t{cluster_labels[i]}\n")
                    del loss, model, th
                res_table.add_row(f"{gname} {clustering}{k} {nlayers}l {pname}", S)
                if gt_idx_label_to_node is not None:
                    # save embs
                    np.save(f"{dataset}_{gname}_{clustering}{k}_{nlayers}l_{pname}_embs.npy", node_new_features)
                # plot_clusters(node_new_features, features.numpy(), cluster_labels, f"{gname} VAMB {nlayers}l {pname}")
    print("Internal results, use AMBER for full evaluation")
    res_table.show()
