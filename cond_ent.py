import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score

from utils import load_preprocessed, matrix_sparsity_info


def conditional_entropy(site1, site2):
	site1 = np.array(site1)
	site2 = np.array(site2)

	if len(site1) != len(site2):
		raise ValueError('site1 and site2 should be same length')

	# make probability tables
	where_s1_0 = (site1 == 0)
	num_s1_0 = where_s1_0.sum()
	where_s1_1 = (site1 == 1)
	num_s1_1 = where_s1_1.sum()

	if num_s1_0 == 0:
		p_0_given_0 = .5
		p_1_given_0 = .5
	else:
		p_0_given_0 = np.count_nonzero(site2[where_s1_0] == 0) / num_s1_0
		p_1_given_0 = np.count_nonzero(site2[where_s1_0] == 1) / num_s1_0

	if num_s1_1 == 0:
		p_0_given_1 = .5
		p_1_given_1 = .5
	else:
		p_0_given_1 = np.count_nonzero(site2[where_s1_1] == 0) / num_s1_1
		p_1_given_1 = np.count_nonzero(site2[where_s1_1] == 1) / num_s1_1

	ent_given_0 = stats.entropy([p_0_given_0, p_1_given_0])
	ent_given_1 = stats.entropy([p_0_given_1, p_1_given_1])

	return (num_s1_0 * ent_given_0 + num_s1_1 * ent_given_1) / (num_s1_0 + num_s1_1)


def max_conditional_entropy(site1, site2):
	res = (conditional_entropy(site1, site2), conditional_entropy(site2, site1))
	return max(res)


def get_connections(site1, site2):
	''' return mask of rows which are fragments common to both sites '''
	return (~np.isnan(site1)) & (~np.isnan(site2))


if __name__ == '__main__':
	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	# get undirected edge weights
	min_connections = 2

	edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	dir_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

	for i, j in tqdm(zip(*np.tril_indices(fragments.shape[1], k=-1)), 
					desc='edges',
					total=((fragments.shape[1] ** 2) / 2 - fragments.shape[1])):

		# check if two sites are connected by enough fragments
		connections = get_connections(fragments[:,i], fragments[:,j])

		if connections.sum() < min_connections:
			continue

		# undirected edges
		corr = max_conditional_entropy(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)

		edges[i,j] = corr
		edges[j,i] = corr

		# directed edges
		dir_edges[i,j] = conditional_entropy(fragments[:,i][connections], 
											 fragments[:,j][connections])
		dir_edges[j,i] = conditional_entropy(fragments[:,j][connections], 
											 fragments[:,i][connections])

	# get average of edges
	edges_mean = edges.mean(axis=1)
	edges_mean = edges_mean.filled(edges_mean.mean())

	out_mean = dir_edges.mean(axis=1)
	out_mean = out_mean.filled(out_mean.mean())

	np.save('data/results/cond_ent_out_5M.npy', out_mean)

	in_mean = dir_edges.mean(axis=0)
	in_mean = in_mean.filled(in_mean.mean())

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			edges_mean,
			out_mean,
			in_mean,
			variant_labels
		)),
		columns=['max', 'out', 'in', 'label'],
	)

	sns.displot(df, x='max', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='out', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='in', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()

	# box plots
	sns.boxplot(
		x=(['max'] * len(variant_labels) 
			+ ['out'] * len(variant_labels)
			+ ['in'] * len(variant_labels)),
		y=np.hstack((edges_mean, out_mean, in_mean)), 
		hue=np.hstack((variant_labels, variant_labels, variant_labels))
	)
	plt.show()

	# plot precision recall and roc curves
	precision, recall, _ = precision_recall_curve(
		variant_labels, edges_mean, pos_label=0)
	fpr, tpr, _ = roc_curve(variant_labels, edges_mean, pos_label=0)

	print('AUC max: {}'.format(roc_auc_score(variant_labels, -edges_mean)))

	out_precision, out_recall, _ = precision_recall_curve(
		variant_labels, out_mean, pos_label=0)
	out_fpr, out_tpr, _ = roc_curve(variant_labels, out_mean, pos_label=0)

	print('AUC out: {}'.format(roc_auc_score(variant_labels, -out_mean)))

	in_precision, in_recall, _ = precision_recall_curve(
		variant_labels, in_mean, pos_label=0)
	in_fpr, in_tpr, _ = roc_curve(variant_labels, in_mean, pos_label=0)

	print('AUC in: {}'.format(roc_auc_score(variant_labels, -in_mean)))

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(recall, precision, label='max')
	plt.plot(out_recall, out_precision, label='out')
	plt.plot(in_recall, in_precision, label='in')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1.01)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(fpr, tpr, label='max')
	plt.plot(out_fpr, out_tpr, label='out')
	plt.plot(in_fpr, in_tpr, label='in')
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()