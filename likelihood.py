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


np.set_printoptions(edgeitems=5, linewidth=1000)


def frag_mat_likelihood(frags, probs, h1, h2):
	row_likelihoods = []

	for frag_row in range(frags.shape[0]):
		row_likelihoods.append(
			row_likelihood(frags[frag_row], probs[frag_row], h1, h2)
		)

	return np.prod(row_likelihoods)


def row_likelihood(frags, probs, h1, h2):
	return (
		(row_likelihood_one_h(frags, probs, h1) 
			+ row_likelihood_one_h(frags, probs, h2))
		/ 2
	)


def row_likelihood_one_h(frags, probs, h):
	row_probs = []

	for read, phred_prob, h_val in zip(frags, probs, h):
		if np.isnan(read):
			row_probs.append(1)
		else:
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


def gen_H_2(H_len, monitor_ind):
	'''
	Generator that yields all needed H matrices to test monitor_ind
	as (2, H_len) dim numpy arrays. Yields a tuple where the first 
	element is the H matrix and the second element is:
		0 if monitor ind in the generated H is 0/0
		1 if 1/1
		2 if 0/1
	'''
	for hap in product([0,1,2,3], repeat=H_len):

		if hap[monitor_ind] == 3:
			continue

		this_H = []

		for site in hap:
			if site == 0:
				this_H.append((0,0))
			elif site == 1:
				this_H.append((0,1))
			elif site == 2:
				this_H.append((1,1))
			else:
				this_H.append((1,0))

		yield np.array(this_H).T, hap[monitor_ind]


def site_prob(haplotype_pair, alpha):
	''' Return prior probability of a haplotype pair '''
	if haplotype_pair[0] == 0 and haplotype_pair[1] == 0:
		return 1 - alpha
	elif haplotype_pair[0] == 1 and haplotype_pair[1] == 1:
		return alpha / 2
	else:
		return alpha / 4


def site_zygosity_probabilities(fragments, probs, site_ind, alpha, progress_bar=False):
	'''
	Return (P(0/0), P(0/1), P(1/1)) for site in given matrices 
	by using sumation of probabilities with all posible H.
	'''
	window_size = fragments.shape[1]
	gt_probs = [0., 0., 0.]

	for H, zygosity in tqdm(gen_H_2(window_size, site_ind), 
							total=3 * 4 ** (window_size-1),
							disable=not progress_bar):
		obs_prob = frag_mat_likelihood(
						fragments, 
						probs, 
						H[0], 
						H[1]
					)
		H_prior = H_prob(H, alpha)

		gt_probs[zygosity] += obs_prob * H_prior

	return gt_probs


def mp_run_zygosity_probabilities(args):
	''' 
	Unpacks args when running zygosity_probabilities with multiprocessing.
	'''
	return site_zygosity_probabilities(args[0], args[1], 0, args[2])


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

	def gen_cols(self, fragments, probs, to_test_inds):
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

			yield fragments[:, cols], probs[:, cols], self.alpha


class closest_cols_generator():
	def __init__(self, alpha, window_size, hetero_inds):
		self.alpha = alpha
		self.window_size = window_size
		self.hetero_inds = np.array(hetero_inds)

	def gen_cols(self, fragments, probs, to_test_inds):
		for i in to_test_inds:
			cols = [i]

			others = (np.abs(self.hetero_inds - i)).argsort()[:self.window_size]
			others_inds = self.hetero_inds[others]
			others_inds = others_inds[others_inds != i][:self.window_size-1]
			cols = cols + others_inds.tolist()

			yield fragments[:, cols], probs[:, cols], self.alpha

	def gen_cols_inds(self, fragments, probs, to_test_inds):
		for i in to_test_inds:
			cols = [i]

			others = (np.abs(self.hetero_inds - i)).argsort()[:self.window_size]
			others_inds = self.hetero_inds[others]
			others_inds = others_inds[others_inds != i][:self.window_size-1]
			cols = cols + others_inds.tolist()

			yield cols


def zygosity_probabilities(fragments, qualities, to_test_inds, site_data, 
									hetero_inds, alpha, window_size, 
									n_processes=None, return_probs_mat=False):
	''' Checks given inds using closest of predicted hetero sites '''
	data_gen = closest_cols_generator(alpha, window_size, hetero_inds)
	
	if n_processes is None:
		n_processes = psutil.cpu_count(logical=False)
		print('Using {} processes'.format(n_processes))

	# convert qualities to probabilities
	probs = np.power(10, -qualities/10)

	with Pool(n_processes) as p:
		r = list(tqdm(
				p.imap(
					mp_run_zygosity_probabilities, 
					data_gen.gen_cols(fragments, probs, to_test_inds)
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
	site_data['P1'] = r[:, 1] * 2.
	site_data['P2'] = r[:, 2]

	if return_probs_mat:
		return site_data, r
	else:
		return site_data


# Command line tool related below
def get_site_probabilities(inds, fragments, qualities, alpha, 
							progress_bar=False):
	'''
	Return a site's (P(0/0), P(0/1), P(1/1)) given the site's index
	and the indices of other sites to compare it with (where 1 + 
	len(other_sites) is equivalent to the 'window size').

	Args:
		inds: list of indices to check, where inds[0] is the site being
			evaluated for. e.g. [site_index, other_ind, other_ind, ...]
	'''
	window_size = len(inds)

	probs = [0., 0., 0.]

	frags_at_inds = fragments[:, inds]
	quals_at_inds = qualities[:, inds]

	for H, zygosity in tqdm(gen_H_2(window_size, 0),
							desc="Checking possible genotypes",
							total=3 * 4 ** (window_size-1),
							disable=not progress_bar):
		obs_prob = frag_mat_likelihood(
						frags_at_inds, 
						quals_at_inds, 
						H[0], 
						H[1]
					)
		H_prior = H_prob(H, alpha)

		probs[zygosity] += obs_prob * H_prior

	return probs


def run_cl_site_probs(fragments_path, longshot_vcf_path, site_inds, alpha, 
						progress_bar=True):
	''' Runs site probability cl tool '''
	from utils import load_longshot_data

	print("Loading Longshot output")
	df, fragments, qualities = load_longshot_data(fragments_path, longshot_vcf_path)

	if not progress_bar:
		print("Checking possible genotypes")
	
	probs = get_site_probabilities(site_inds, fragments, qualities, alpha, True)

	print("P(0/0):\t{}\tP(0/1):\t{}\tP(1/1):\t{}".format(*probs))


def main():
	pass


if __name__ == '__main__':
	from_preprocessed = True
	save_preprocessed = True
	prepro_save_dir = 'data/preprocessed/chr20_1-1M'

	alpha = 0.001 			# user defined for site genotype priors
	window_size = 1

	save_results = True
	save_path = 'data/results/1M_predicitions_v2_1_w{}_a{}.tsv'.format(
		window_size, str(alpha).split('.')[1])

	fragments_path='data/fragments/chr20_1-1M/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-1M/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	giab_bed_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.bed'
	site_data_save_path='data/preprocessed/1M_site_data.tsv'

	# Load either preprocessed data or data from vcfs, bed, and fragments.txt
	if from_preprocessed:
		print('Loading preprocessed data')
		df = pd.read_csv(os.path.join(prepro_save_dir, 'df.csv'))
		fragments = np.load(os.path.join(prepro_save_dir, 'fragments.npy'))
		qualities = np.load(os.path.join(prepro_save_dir, 'qualities.npy'))
	else:
		print('Loading data')
		from utils import load_full_data

		df, fragments, qualities = load_full_data(
			fragments_path, 
			longshot_vcf_path, 
			ground_truth_vcf_path,
			giab_bed_path, 
			save_path=site_data_save_path)

		df = df[df.pos < 1000000] 		# change this with size of Longshot res used

		if save_preprocessed:
			df.to_csv(os.path.join(prepro_save_dir, 'df.csv'), index=False)
			np.save(
				os.path.join(prepro_save_dir, 'fragments.npy'),
				fragments,
				fix_imports=False
			)
			np.save(
				os.path.join(prepro_save_dir, 'qualities.npy'),
				qualities,
				fix_imports=False
			)

	# df = df.sample(10)

	to_test = np.where(df.in_bed)[0]
	
	res, probs_mat = zygosity_probabilities(
		fragments,
		qualities,
		to_test,
		df[['site_ind', 'chrom', 'pos']].iloc[to_test].reset_index(drop=True),
		df[df.ls_hmm_pred_genotype == 1].site_ind.values.astype(int),
		alpha,
		window_size,
		return_probs_mat=True
	)

	res_df = pd.merge(df, res, how='left', on=['site_ind', 'chrom', 'pos'])

	if save_results:
		res_df.to_csv(save_path, na_rep='', sep='\t', index=False)
