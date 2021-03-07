import random
import itertools
import collections

import numpy as np
from sklearn.metrics import classification_report
from scipy.stats.stats import pearsonr 
import networkx as nx
import matplotlib.pyplot as plt

from utils import *


def remove_empty_homozygous_sites(fragments, variant_labels):
    n_variants = fragments.shape[1]
    to_remove = []
    # Remove homozygous sites
    for j in np.arange(n_variants):
        x = fragments[:,j]; x = x[~np.isnan(x)]
        if len(x) > 0 and (all(x == x[0])):
            to_remove.append(j)
    # Remove sites wihtout reads
    reads_per_var = np.sum(np.isfinite(fragments), axis=0)
    to_remove.extend(list(np.where(reads_per_var==0)[0]))
    
    # Exclude the sites from fragments and variant labels
    fragments = np.delete(fragments, to_remove, axis=1)
    variant_labels = np.delete(variant_labels, to_remove, axis=0)
    return(fragments, variant_labels)


def build_graph(fragments):
    n_variants = fragments.shape[1]
    G = nx.Graph()
    for i,j in itertools.combinations(np.arange(n_variants),2):
        col_i = fragments[:,i]
        col_j = fragments[:,j]

        valid_is = np.where(np.isfinite(fragments[:,i]))[0]
        valid_js = np.where(np.isfinite(fragments[:,j]))[0]
        valid_ijs = np.array(list(set(valid_is).intersection(set(valid_js))))
        if len(valid_ijs) > 0:
            frag_i = col_i[valid_ijs]
            frag_j = col_j[valid_ijs]
            not_frag_j = (~frag_j.astype(bool)).astype(int)

            diffs = min(sum(chaine1 != chaine2 for chaine1, chaine2 in zip(frag_i, frag_j)),
                        sum(chaine1 != chaine2 for chaine1, chaine2 in zip(frag_i, not_frag_j)))

            edge_weight = diffs/len(valid_ijs)
            G.add_edge(i, j, weight=edge_weight)
    return(G)

def prefilter_true_variant(G):
    all_nodes = set(G.nodes())
    volumes = [ nx.volume(G, {w}, "weight")/nx.volume(G, {w}) for w in all_nodes]
    median_volume = np.median(volumes)
    predicted_true_variants = np.where(volumes < median_volume)[0]
    return(predicted_true_variants)

def greedy_cut(G, prefiltered_true_variants = []):
    G.remove_nodes_from(prefiltered_true_variants)
    all_nodes = set(G.nodes)
    all_edges = list(G.edges)
    
    # 1. Heuristic to find a cut in whith the highly negative weight edges do not cross the cut
    S1 = set()
    S2 = set()
    u, v = random.choice(all_edges)
    S1.add(u); S2.add(v)
    
    while True:
        S = S1.union(S2)
        C = all_nodes.difference(S)
        if len(C) == 0:
            break
        A = {}
        for w in C:
            A[w] = nx.cut_size(G, {w}, S1, weight="weight") - nx.cut_size(G, {w}, S2, weight="weight")        
        maxkey = max(A, key=lambda x: abs(A[x]))
        rn = random.uniform(0, 1)
        if (A[maxkey] < 0) or ((A[maxkey] == 0) and (rn < 0.5)):
            S1.add(maxkey)
        else:
            S2.add(maxkey)
            
            
    # 2. Greedy max-cut algorithm
    while True:
        A1 = {}; A2 = {}
        old_cut = nx.cut_size(G, S1, S2, weight="weight")
        for w in S1:
            A1[w] = nx.cut_size(G, {w}, S1, weight="weight") - nx.cut_size(G, {w}, S2, weight="weight")       
        for w in S2:
            A2[w] = nx.cut_size(G, {w}, S1, weight="weight") - nx.cut_size(G, {w}, S2, weight="weight")       

        V1 = [k for k, v in A1.items() if v > 0]
        V2 = [k for k, v in A2.items() if v < 0]

        for w in V1:
            S2.add(w)
            S1.remove(w)

        for w in V2:
            S1.add(w)
            S2.remove(w)

        new_cut = nx.cut_size(G, S1, S2, weight="weight")
        if new_cut <= old_cut:
            break
    S = S1 if len(S1) < len(S2) else S2
    return(S)


if __name__ == '__main__':

	fragments_path='data/fragments/chr20_1-500K/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-500K/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	giab_bed_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.bed'

	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	print('All fragments:')
	matrix_sparsity_info(fragments, print_info=True)

	# get real/false variant labels
	variant_labels, site_mask, ls_callset, callset = get_true_variants(
	    longshot_vcf_path=longshot_vcf_path,
	    ground_truth_vcf_path=ground_truth_vcf_path,
	    giab_bed_path=giab_bed_path,
	    return_vcfs=True
	)

	# mask these with site_mask
	fragments = fragments[:, site_mask]
	qualities = qualities[:, site_mask]

	print('\nMasked fragments:')
	matrix_sparsity_info(fragments, print_info=False)

	# start experiment
	fragments, variant_labels = remove_empty_homozygous_sites(fragments, variant_labels)
	G = build_graph(fragments)
	prefiltered_true_variants = prefilter_true_variant(G)
	predicted_false_variants = greedy_cut(G, prefiltered_true_variants)

	# evaluation
	pred_labels = np.ones_like(variant_labels)
	pred_labels[list(predicted_false_variants)] = 0
	cr = classification_report(variant_labels, pred_labels)
	print('\nPredict all false:\n' + cr)



