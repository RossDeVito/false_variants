import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats
from scipy.spatial import distance

from sklearn.metrics import precision_recall_curve, roc_curve

from utils import *


def hellinger_distance(p, q):
	return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / np.sqrt(2)


def additive_smoothing(observations, alpha):
	''' 
	Performs additive smoothing where rows or observations are individual
	observed distributions and alpha is the smoothing parameter
	'''
	obs = observations + alpha
	return obs / (1 + alpha * len(obs[0]))


if __name__ == '__main__':
	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	# homogeneous sites should be near 100-0 ratio, hetro 50-50
	frags_ma = np.ma.masked_invalid(fragments)

	zero_count = np.sum(frags_ma == 0, axis=0).data
	one_count = np.sum(frags_ma == 1, axis=0).data

	# find number of reads at each site and remove if under threshold
	site_depth = zero_count + one_count
	enough_site_depth = site_depth > 5
	print("ignore {} sites for insufficient number of reads".format(
		(~enough_site_depth).sum()
	))

	fragments = fragments[:, enough_site_depth]
	zero_count = zero_count[enough_site_depth]
	one_count = one_count[enough_site_depth]
	site_depth = site_depth[enough_site_depth]
	variant_labels = variant_labels[enough_site_depth]

	zero_ratio = zero_count / site_depth
	one_ratio = one_count / site_depth

	# plot ratios relation to labels
	df = pd.DataFrame(
		np.column_stack((
			np.append(zero_ratio, one_ratio), 
			np.append(variant_labels, variant_labels)
		)),
		columns=['ratio', 'label'],
	)

	sns.displot(df, x='ratio', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()

	# use KL divergence or hellinger distance to identify false variants
	likely_true_dists = np.array([[.5, .5], [1, 0], [0, 1]])
	sample_dists = np.column_stack((zero_ratio, one_ratio))

	# smooth

	ltd_smoothed = additive_smoothing(likely_true_dists, .1) 
	sd_smoothed = additive_smoothing(sample_dists, .1)

	hellinger = np.empty((sample_dists.shape[0], likely_true_dists.shape[0]))
	hellinger_smoothed = np.empty((sample_dists.shape[0], likely_true_dists.shape[0]))
	euc = np.empty((sample_dists.shape[0], likely_true_dists.shape[0]))
	kl_div = np.empty((sample_dists.shape[0], likely_true_dists.shape[0]))
	js = np.empty((sample_dists.shape[0], likely_true_dists.shape[0]))

	for samp_ind in tqdm(range(hellinger.shape[0]), desc='Getting distances'):
		for dist_ind in range(hellinger.shape[1]):
			# kld[samp_ind, dist_ind] = stats.entropy(sample_dists[samp_ind], 
			# 										likely_true_dists[dist_ind])
			hellinger[samp_ind, dist_ind] = hellinger_distance(
				sample_dists[samp_ind], likely_true_dists[dist_ind]
			)
			hellinger_smoothed[samp_ind, dist_ind] = hellinger_distance(
				sd_smoothed[samp_ind], ltd_smoothed[dist_ind]
			)
			euc[samp_ind, dist_ind] = distance.euclidean(
				sample_dists[samp_ind], likely_true_dists[dist_ind]
			)
			kl_div[samp_ind, dist_ind] = stats.entropy(
				sd_smoothed[samp_ind], ltd_smoothed[dist_ind]
			)
			js[samp_ind, dist_ind] = distance.jensenshannon(
				sd_smoothed[samp_ind], ltd_smoothed[dist_ind]
			)

	best_hellinger = np.min(hellinger, axis=1)
	best_euc = np.min(euc, axis=1)
	best_hellinger_s = np.min(hellinger_smoothed, axis=1)
	best_kl = np.min(kl_div, axis=1)
	best_js = np.min(js, axis=1)

	# both dists
	plt.subplots()
	plt.subplot(1,4,1)
	sns.boxplot(x=variant_labels, y=best_hellinger, hue=variant_labels)
	plt.title('Hellinger')
	plt.subplot(1,4,2)
	sns.boxplot(x=variant_labels, y=best_euc, hue=variant_labels)
	plt.title('Euclidean')
	plt.subplot(1,4,3)
	sns.boxplot(x=variant_labels, y=best_hellinger_s, hue=variant_labels)
	plt.title('Hellinger (smoothed)')
	plt.subplot(1,4,4)
	sns.boxplot(x=variant_labels, y=best_kl, hue=variant_labels)
	plt.title('KL Divergence (smoothed)')
	plt.show()
			
	# plot precision recall and roc curves
	precision, recall, _ = precision_recall_curve(
		variant_labels, best_hellinger, pos_label=0)
	fpr, tpr, _ = roc_curve(variant_labels, best_hellinger, pos_label=0)

	euc_precision, euc_recall, _ = precision_recall_curve(
		variant_labels, best_euc, pos_label=0)
	euc_fpr, euc_tpr, _ = roc_curve(variant_labels, best_euc, pos_label=0)

	precision_s, recall_s, _ = precision_recall_curve(
		variant_labels, best_hellinger_s, pos_label=0)
	fpr_s, tpr_s, _ = roc_curve(variant_labels, best_hellinger_s, pos_label=0)

	kl_precision, kl_recall, _ = precision_recall_curve(
		variant_labels, best_kl, pos_label=0)
	kl_fpr, kl_tpr, _ = roc_curve(variant_labels, best_kl, pos_label=0)

	js_precision, js_recall, _ = precision_recall_curve(
		variant_labels, best_js, pos_label=0)
	js_fpr, js_tpr, _ = roc_curve(variant_labels, best_js, pos_label=0)

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(recall, precision, label='Hellinger distance')
	plt.plot(euc_recall, euc_precision, '--', label='Euclidian distance')
	plt.plot(recall_s, precision_s, '-.', label='Hellinger distance smoothed')
	plt.plot(kl_recall, kl_precision, ':', label='KL divergence smoothed')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(fpr, tpr, label='Hellinger distance')
	plt.plot(euc_fpr, euc_tpr, '--', label='Euclidian distance')
	plt.plot(fpr_s, tpr_s, '-.', label='Hellinger distance smoothed')
	plt.plot(kl_fpr, kl_tpr, ':', label='KL divergence smoothed')
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)
	
	plt.show()