import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils import load_preprocessed, matrix_sparsity_info


np.set_printoptions(edgeitems=5, linewidth=1000)


def frag_mat_likelihood(frags, quals, h1, h2):
	row_likelihoods = []

	for frag_row in range(frags.shape[0]):
		row_likelihoods.append(
			row_likelihood(frags[frag_row], quals[frag_row], h1, h2)
		)

	return np.prod(row_likelihoods)


def frag_mat_likelihood_complement(frags, quals, h):
	''' not needed '''
	row_likelihoods = []

	for frag_row in range(frags.shape[0]):
		row_likelihoods.append(
			row_likelihood(frags[frag_row], quals[frag_row], h, (h == 0).astype(int))
		)

	return np.prod(row_likelihoods)


def row_likelihood(frags, quals, h1, h2):
	return (
		(row_likelihood_one_h(frags, quals, h1) 
			+ row_likelihood_one_h(frags, quals, h2))
		/ 2
	)


def row_likelihood_one_h(frags, quals, h):
	row_probs = []

	for read, phred, h_val in zip(frags, quals, h):
		if np.isnan(read):
			row_probs.append(1)
		else:
			phred_prob = 10 ** (-phred / 10)
			matches_h = int(read == h_val)

			row_probs.append(
				matches_h * (1 - phred_prob) + (1 - matches_h) * phred_prob
			)
	
	return np.prod(row_probs)


def H_prob(H, alpha):
	'''
	Prior probability of H, which consists of two binary haplotypes

	Args:
		H:	Two binary haplotypes represented as a (2, haplotype_length) dim
				numpy array
		alpha:	alpha value in pair probability calculations

	Returns prior probability of H
	'''
	prob = 1.0

	for site_ind in range(H.shape[1]):
		prob *= site_prob(H[:, site_ind], alpha)

	return prob


def site_prob(haplotype_pair, alpha):
	''' Return prior probability of a haplotype pair '''
	if haplotype_pair[0] == 0 and haplotype_pair[1] == 0:
		return 1 - alpha
	elif haplotype_pair[0] == 1 and haplotype_pair[1] == 1:
		return alpha / 2
	else:
		return alpha / 4


if __name__ == '__main__':
	alpha = 0.001 			# user defined for site genotype priors

	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)

	frag_row = 1271 #1273

	window_size = 5
	end_ind = 756 #500 for nans

	cols = range(end_ind - window_size, end_ind)

	frags = fragments[frag_row, cols]
	quals = qualities[frag_row, cols]
	h0 = np.array([1, 1, 0, 1, 1])
	h1 = np.array([0, 1, 0, 1, 0])
	h2 = np.array([0, 0, 1, 0, 0])

	h3 = np.array([0, 0, 0, 0, 0])

	f2 = fragments[frag_row+2, cols]

	print(row_likelihood_one_h(frags, quals, h0))
	print(row_likelihood_one_h(frags, quals, h1))
	print(row_likelihood_one_h(frags, quals, h2))
	print(row_likelihood_one_h(frags, quals, h3))
	print()
	print(frag_mat_likelihood(fragments[:, cols], qualities[:, cols], h0, h3))
	print(frag_mat_likelihood(fragments[:, cols], qualities[:, cols], h0, h0))
	print(frag_mat_likelihood(fragments[:, cols], qualities[:, cols], h3, h3))
	print(frag_mat_likelihood(fragments[:, cols], qualities[:, cols], h0, h1))
	print()

	H = np.vstack((h0,h3))

	prior_prob = H_prob(H, alpha)
	print(prior_prob)