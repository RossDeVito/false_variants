import os
import psutil
import warnings
from itertools import product
from multiprocessing import Pool

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve

from utils import load_data, matrix_sparsity_info


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


def gen_H(H_len, monitor_ind=None):
	'''
	Generator that yields all posible H matrices as (2, H_len) dim 
	numpy arrays. If monitor_ind is not None, instead yields a tuple
	where the first element is the H matrix and the second element is:
		0 if monitor ind in the generated H is 0|0
		1 if 1|1
		2 if 0|1
		3 if 1|0
	'''
	for hap in product([0,1,2,3], repeat=H_len):
		this_H = []

		for site in hap:
			if site == 0:
				this_H.append((0,0))
			elif site == 1:
				this_H.append((1,1))
			elif site == 2:
				this_H.append((0,1))
			else:
				this_H.append((1,0))
		
		if monitor_ind is not None:
			yield np.array(this_H).T, hap[monitor_ind]
		else:
			yield np.array(this_H).T


def site_prob(haplotype_pair, alpha):
	''' Return prior probability of a haplotype pair '''
	if haplotype_pair[0] == 0 and haplotype_pair[1] == 0:
		return 1 - alpha
	elif haplotype_pair[0] == 1 and haplotype_pair[1] == 1:
		return alpha / 2
	else:
		return alpha / 4


def zygosity_probabilities(fragments, qualities, site_ind, alpha, progress_bar=False):
	'''
	Return (P(0|0), P(1|1), P(0|1), P(1|0)) for site in given matrices 
	by using sumation of probabilities with all posible H.
	'''
	window_size = fragments.shape[1]
	probs = [0., 0., 0., 0.]

	for H, zygosity in tqdm(gen_H(window_size, site_ind), 
							total=4 ** window_size,
							disable=not progress_bar):
		obs_prob = frag_mat_likelihood(
						fragments, 
						qualities, 
						H[0], 
						H[1]
					)
		H_prior = H_prob(H, alpha)

		probs[zygosity] += obs_prob * H_prior

	return probs


def mp_run_zygosity_probabilities(args):
	return zygosity_probabilities(args[0], args[1], 0, args[2])


def all_zygosity_probabilities(fragments, qualities, alpha, window_size):
	'''
	Return arrays of P(homozygous) and P(heterozygous) based on trying
	all possible H for a window arouns the site whose probability is given.
	'''
	probs = []

	for i in tqdm(range(fragments.shape[1])):
		cols = [i]
		one_dir = window_size // 2
		is_odd = window_size % 2

		if i - one_dir >= 0 and i + one_dir < fragments.shape[1]:
			cols.extend(range(i+1, i+is_odd+one_dir))
			cols.extend(range(i-one_dir, i))
		elif i - one_dir < 0:
			over = abs(i - one_dir)
			these_cols = list(range(i-one_dir+over, i+is_odd+one_dir+over))
			these_cols.remove(i)
			cols.extend(these_cols)
		elif i + one_dir >= fragments.shape[1]:
			over = i + one_dir - fragments.shape[1]
			these_cols = list(range(i-one_dir-over-is_odd, fragments.shape[1]))
			these_cols.remove(i)
			cols.extend(these_cols)
		else:
			raise RuntimeError()

		col_probs = zygosity_probabilities(
			fragments[:, cols],
			qualities[:, cols],
			0,
			alpha)

		probs.append(col_probs)

	return np.vstack(probs)


class column_generator():
	def __init__(self, alpha, window_size):
		self.alpha = alpha
		self.window_size = window_size
		self.one_dir = window_size // 2
		self.is_odd = window_size % 2

	def gen_cols(self, fragments, qualities, to_test_inds):
		for i in to_test_inds:
			cols = [i]

			if i - self.one_dir >= 0 and i + self.one_dir < fragments.shape[1]:
				cols.extend(range(i+1, i+self.is_odd+self.one_dir))
				cols.extend(range(i-self.one_dir, i))
			elif i - self.one_dir < 0:
				over = abs(i - self.one_dir)
				these_cols = list(range(i-self.one_dir+over, i+self.is_odd+self.one_dir+over))
				these_cols.remove(i)
				cols.extend(these_cols)
			elif i + self.one_dir >= fragments.shape[1]:
				over = i + self.one_dir - fragments.shape[1]
				these_cols = list(range(i-self.one_dir-over-self.is_odd, fragments.shape[1]))
				these_cols.remove(i)
				cols.extend(these_cols)
			else:
				raise RuntimeError()

			yield fragments[:, cols], qualities[:, cols], self.alpha



def mp_all_zygosity_probabilities(fragments, qualities, to_test_inds, site_data, 
									alpha, window_size, n_processes=None):
	data_gen = column_generator(alpha, window_size)
	
	if n_processes is None:
		n_processes = psutil.cpu_count(logical=False)

	with Pool(n_processes) as p:
		r = list(tqdm(
				p.imap(
					mp_run_zygosity_probabilities, 
					data_gen.gen_cols(fragments, qualities, to_test_inds)
				),
				total=len(to_test_inds), 
				desc='Running tests',
				smoothing=0.1
			))

	r = np.vstack(r)

	site_data['window_size'] = window_size
	site_data['window_size'].astype(int)
	site_data['alpha'] = alpha
	site_data['P0'] = r[:, 0]
	site_data['P1'] = r[:, 2:].sum(axis=1)
	site_data['P2'] = r[:, 1]

	return site_data


if __name__ == '__main__':
	alpha = 0.001 			# user defined for site genotype priors
	window_size = 5
	save_results = True
	save_path = 'data/results/1M_predicitions_w{}_a{}.tsv'.format(
		window_size, str(alpha).split('.')[1])

	fragments_path='data/fragments/chr20_1-1M/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-1M/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	giab_bed_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.bed'
	site_data_save_path='data/preprocessed/1M_site_data.tsv'

	print('Loading data')
	df, fragments, qualities = load_data(
		fragments_path, 
		longshot_vcf_path, 
		ground_truth_vcf_path,
		giab_bed_path, 
		save_path=site_data_save_path)

	# df = df.sample(10)

	to_test = np.where(df.in_bed)[0]
	
	res = mp_all_zygosity_probabilities(
		fragments,
		qualities,
		to_test,
		df[['site_ind', 'chrom', 'pos']].iloc[to_test].reset_index(drop=True),
		alpha,
		window_size
	)

	res_df = pd.merge(df, res, how='left', on=['site_ind', 'chrom', 'pos'])

	if save_results:
		res_df.to_csv(save_path, na_rep='', sep='\t', index=False)


	# variant_labels = variant_labels[:n]
	

	# all_probs = mp_all_zygosity_probabilities(
	# 	fragments, 
	# 	qualities,
	# 	alpha,
	# 	window_size,
	# 	n_processes=4)

	# p_00 = all_probs[:, 0]
	# p_11 = all_probs[:, 1]
	# p_01 = all_probs[:, 2]
	# p_10 = all_probs[:, 3]

	# pred = p_01 + p_10 >= p_00 + p_11

	# print(classification_report(variant_labels, pred, labels=[0,1],
	# 							target_names=['false variant', 'true variant']))
	# cm = confusion_matrix(variant_labels, pred)
	# print(cm)

	# scores = p_01 + p_10 / p_00 + p_11

	# if save_results:
	# 	np.save('data/results/likelihood_w{}_1M.npy'.format(window_size), scores)
	# 	np.save('data/results/likelihood_w{}_p00_1M.npy'.format(window_size), p_00)
	# 	np.save('data/results/likelihood_w{}_p11_1M.npy'.format(window_size), p_11)
	# 	np.save('data/results/likelihood_w{}_p01_1M.npy'.format(window_size), p_01)
	# 	np.save('data/results/likelihood_w{}_p10_1M.npy'.format(window_size), p_10)

	# precision, recall, threshold = precision_recall_curve(
	# 	variant_labels, -scores, pos_label=0)
	# fpr, tpr, _ = roc_curve(variant_labels, -scores, pos_label=0)

	# plt.subplots()
	# plt.subplot(1,2,1)
	# plt.plot(recall, precision, label='window = {}'.format(window_size))
	# plt.xlabel('false variant recall')
	# plt.xlim(0, 1.01)
	# plt.ylabel('false variant precision')
	# plt.ylim(0, 1.01)
	# plt.legend()
	# plt.title('Precision Recall Curve')
	# plt.gca().set_aspect('equal', adjustable='box')
	# plt.grid(True)

	# plt.subplot(1,2,2)
	# plt.plot(fpr, tpr, label='window = {}'.format(window_size))
	# plt.xlabel('false positive rate (positive label: false variant)')
	# plt.xlim(0, 1.01)
	# plt.ylabel('true positive rate')
	# plt.ylim(0, 1.01)
	# plt.legend()
	# plt.title('ROC Curve')
	# plt.gca().set_aspect('equal', adjustable='box')
	# plt.grid(True)

	# plt.show()
	