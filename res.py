import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, confusion_matrix

from utils import load_preprocessed, matrix_sparsity_info


def get_best_f1(precision, recall, thresholds):
	numerator = 2 * recall * precision
	denom = recall + precision
	f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))

	return np.max(f1_scores), thresholds[np.argmax(f1_scores)]


def get_cm_at_cutoff(cutoff, scores, variant_labels, pos_label=0, normalize=None):
	''' 
	Return confusion matrix and best f1 score (if cutoff is 'f1' else None) 
	'''
	precision, recall, thresholds = precision_recall_curve(
		variant_labels, -scores, pos_label=pos_label
	)

	if cutoff == 'f1':
		best_f1, thresh = get_best_f1(precision, recall, thresholds)

		preds = scores > -thresh

		return confusion_matrix(variant_labels, preds, normalize=normalize), best_f1
	else:
		thresh_ind = np.where(precision > cutoff)[0].min()

		if thresh_ind < len(thresholds):
			thresh = -thresholds[thresh_ind]
		else:
			thresh = 0

		preds = scores > thresh

		return confusion_matrix(variant_labels, preds, normalize=normalize), None

if __name__ == '__main__':
	# load preprocessed data
	_, _, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)

	save_dir = 'data/results'
	
	# # original
	# res_saves = [
	# 	('phi_correlation_5M.npy', 'phi correlation'),
	# 	#('fisher_p05_5M.npy', 'fisher .05'),
	# 	('fisher_p01_5M.npy', 'fisher .01'),
	# 	#('fisher_p005_5M.npy', 'fisher .005'),
	# 	#('chi_p05_5M.npy', 'chi .05'),
	# 	('chi_p01_5M.npy', 'chi .01'),
	# 	#('chi_p005_5M.npy', 'chi .005'),
	# ]

	# # top scores
	# res_saves = [
	# 	('phi_correlation_5M.npy', 'phi correlation'),
	# 	('adj_rand_index_5M.npy', 'adj. rand'),
	# 	('norm_mutual_info_5M.npy', 'NMI'),
	# 	('homogeneity_out_5M.npy', 'homo. out'),
	# ]

	# results = [(np.load(os.path.join(save_dir, s[0])), s[1]) for s in res_saves]

	# for i, r in enumerate(results):
	# 	print(r[1])

	# 	# # flip
	# 	# if r[1] == 'phi correlation':
	# 	# 	results[i] = (-r[0], r[1])

	# 	get_best_f1(*precision_recall_curve(variant_labels, -results[i][0], pos_label=0))
	# 	print()

	# # box plots
	# x = []
	# y = []
	# hue = []

	# for r in results:
	# 	x.extend([r[1]] * len(variant_labels))
	# 	y.extend(r[0].tolist())
	# 	hue.extend(variant_labels.tolist())

	# sns.boxplot(
	# 	x=x,
	# 	y=y, 
	# 	hue=hue
	# )
	# plt.show()

	# # get confusion matrices at different cuttoffs

	# cutoffs = [.999, .99, .985, .9825, .98, .975, 'f1']

	# # plot all at cutoffs
	# fig, axs = plt.subplots(len(results), len(cutoffs))

	# for j, r in enumerate(results):
	# 	for i, c in enumerate(cutoffs):
	# 		cm, best_f1 = get_cm_at_cutoff(c, r[0], variant_labels)
			
	# 		ax = axs[j, i]
	# 		sns.heatmap(cm, annot=True, ax=ax, fmt='g', cbar=False)
	# 		ax.set_xlabel('Predicted labels')
	# 		if i == 0:
	# 			ax.set_ylabel(r[1], rotation=90, size='large')
	# 		else:
	# 			ax.set_ylabel('True labels')
	# 		if c == 'f1':
	# 			ax.set_title('best f1 (prec = {:.3})'.format(best_f1))
	# 		else:
	# 			ax.set_title('precision > {:.4}'.format(c)) 
	# 		ax.xaxis.set_ticklabels(['false', 'true'])
	# 		ax.yaxis.set_ticklabels(['true', 'false'])
	# 		ax.set_aspect("equal")

	# fig.suptitle("Confusion Matrices at False Variant Precision Thresholds")
	# plt.show()

	# plot test based at different p-vals
	n_p_vals = 5
	normalize = 'true'

	test_saves = [
		[
			('fisher_p1_5M.npy', .1),
			('fisher_p05_5M.npy', .05),
			('fisher_p01_5M.npy', .01),
			('fisher_p005_5M.npy', .005),
			('fisher_p001_5M.npy', .001),
			'Fisher'
		], [
			('chi_p1_5M.npy', .1),
			('chi_p05_5M.npy', .05),
			('chi_p01_5M.npy', .01),
			('chi_p005_5M.npy', .005),
			('chi_p001_5M.npy', .001),
			'Chi^2'
		]
	]

	test_results = []

	for test_res_files in test_saves:
		test_name = test_res_files.pop()
		test_res = [(np.load(os.path.join(save_dir, s[0])), s[1]) for s in test_res_files]

		test_results.append( (test_res, test_name) )

	fig, axs = plt.subplots(len(test_results), n_p_vals)

	for j, r in enumerate(test_results):
		for i, (scores, p_val) in enumerate(r[0]):

			preds = (~(scores == 0)).astype(int)
			cm = confusion_matrix(variant_labels, preds, normalize=normalize)
			
			ax = axs[j, i]
			sns.heatmap(cm, annot=True, ax=ax, fmt='.3g', cbar=False)
			ax.set_xlabel('predicted labels')
			if i == 0:
				ax.set_ylabel(r[1], rotation=90, size='large')
			else:
				ax.set_ylabel('true labels')
			ax.set_title('p-val < {}'.format(p_val)) 
			ax.xaxis.set_ticklabels(['false', 'true'])
			ax.yaxis.set_ticklabels(['true', 'false'])
			ax.set_aspect("equal")

	fig.suptitle("Removing Sites Statistically Independant of All Other Sites (with recall)")
	plt.show()