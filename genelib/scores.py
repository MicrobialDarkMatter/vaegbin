import numpy as np
import scipy

def completeness(reference_markers, genes):
    numerator = 0.0
    for marker_set in reference_markers:
        common = marker_set & genes
        if len(marker_set) > 0:
            numerator += len(common) / len(marker_set)
    return 100 * (numerator / len(reference_markers))


def contamination(reference_markers, genes):
    numerator = 0.0
    for i, marker_set in enumerate(reference_markers):
        inner_total = 0.0
        for gene in marker_set:
            if gene in genes and genes[gene] > 0:
                inner_total += genes[gene] - 1.0
        if len(marker_set) > 0:
            numerator += inner_total / len(marker_set)
    return 100.0 * (numerator / len(reference_markers))

def compute_cluster_score(reference_markers, contig_genes, node_names, node_labels):
    labels_to_nodes =  {i:node_names[node_labels == i].tolist() for i in np.unique(node_labels)}
    results = {}
    for label in labels_to_nodes:
        genes = {}
        for node_name in labels_to_nodes[label]:
            if node_name not in contig_genes:
                # print("missing", node_name)
                continue
            for gene in contig_genes[node_name]:
                if gene not in genes:
                    genes[gene] = 0
                genes[gene] += contig_genes[node_name][gene]
                
        comp = completeness(reference_markers, set(genes.keys()))
        cont = contamination(reference_markers, genes)
        results[label] = {"comp": comp, "cont": cont, "genes": genes}
    return results

def compute_hq(reference_markers, contig_genes, node_names, node_labels, comp_th=90, cont_th=5):
    cluster_stats = compute_cluster_score(reference_markers, contig_genes, node_names, node_labels)
    hq = 0
    positive_clusters = []
    for label in cluster_stats:
        if cluster_stats[label]["comp"] >= comp_th and cluster_stats[label]["cont"] < cont_th:
            hq += 1
            positive_clusters.append(label)
    return hq, positive_clusters


# Label based evaluation from RepBin
# Get precicion
def getPrecision(mat, k, s, total):
    sum_k = 0
    for i in range(k):
        max_s = 0
        for j in range(s):
            if mat[i][j] > max_s:
                max_s = mat[i][j]
        sum_k += max_s
    return sum_k / total


# Get recall
def getRecall(mat, k, s, total, unclassified):
    sum_s = 0
    for i in range(s):
        max_k = 0
        for j in range(k):
            if mat[j][i] > max_k:
                max_k = mat[j][i]
        sum_s += max_k
    return sum_s / (total + unclassified)


# Get ARI
def getARI(mat, k, s, N):
    t1 = 0
    for i in range(k):
        sum_k = 0
        for j in range(s):
            sum_k += mat[i][j]
        t1 += scipy.special.binom(sum_k, 2)
    t2 = 0
    for i in range(s):
        sum_s = 0
        for j in range(k):
            sum_s += mat[j][i]
        t2 += scipy.special.binom(sum_s, 2)
    t3 = t1 * t2 / scipy.special.binom(N, 2)
    t = 0
    for i in range(k):
        for j in range(s):
            t += scipy.special.binom(mat[i][j], 2)
    ari = (t - t3) / ((t1 + t2) / 2 - t3)
    return ari


# Get F1-score
def getF1(prec, recall):
    if prec == 0.0 or recall == 0.0:
        return 0.0
    else:
        return 2 * prec * recall / (prec + recall)


def calculate_overall_prf(cluster_to_contig, contig_to_cluster, node_to_label, label_to_node):
    # calculate how many contigs are in the majority class of each cluster
    total_binned = 0
    # convert everything to ids
    labels = list(label_to_node.keys())
    clusters = list(cluster_to_contig.keys())
    n_pred_labels = len(clusters)
    n_true_labels = len(labels)
    ground_truth_count = len(node_to_label)
    bins_species = [[0 for x in range(n_true_labels)] for y in range(n_pred_labels)]
    for i in contig_to_cluster:
        if i in node_to_label:
            # breakpoint()
            total_binned += 1
            bins_species[clusters.index(contig_to_cluster[i])][labels.index(node_to_label[i])] += 1

    my_precision = getPrecision(bins_species, n_pred_labels, n_true_labels, total_binned)
    my_recall = getRecall(
        bins_species, n_pred_labels, n_true_labels, total_binned, (ground_truth_count - total_binned)
    )
    my_ari = getARI(bins_species, n_pred_labels, n_true_labels, total_binned)
    my_f1 = getF1(my_precision, my_recall)
    return my_precision, my_recall, my_f1, my_ari