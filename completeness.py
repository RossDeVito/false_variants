import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from sklearn.metrics import homogeneity_completeness_v_measure
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score

from utils import load_preprocessed, matrix_sparsity_info


def get_connections(site1, site2):
	''' return mask of rows which are fragments common to both sites '''
	return (~np.isnan(site1)) & (~np.isnan(site2))


if __name__ == '__main__':
	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-500K.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	# get undirected edge weights
	min_connections = 2

	edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	homo_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	comp_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

	for i, j in tqdm(zip(*np.tril_indices(fragments.shape[1], k=-1)), 
					desc='edges',
					total=((fragments.shape[1] ** 2) / 2 - fragments.shape[1])):

		# check if two sites are connected by enough fragments
		connections = get_connections(fragments[:,i], fragments[:,j])

		if connections.sum() < min_connections:
			continue

		# undirected edges
		# i is pred, j true
		homo, comp, v_measure = homogeneity_completeness_v_measure(
			fragments[:,j][connections], 
			fragments[:,i][connections]
		)

		edges[i,j] = v_measure
		edges[j,i] = v_measure

		# directed edges
		homo_edges[i,j] = homo
		homo_edges[j,i] = comp

		comp_edges[i,j] = comp
		comp_edges[j,i] = homo

	# get average of edges
	edges_mean = edges.mean(axis=1)
	edges_mean = edges_mean.filled(edges_mean.mean())

	# get homogeneity
	out_mean = homo_edges.mean(axis=1)
	out_mean = out_mean.filled(out_mean.mean())

	# np.save('data/results/cond_ent_out_5M.npy', out_mean)

	in_mean = homo_edges.mean(axis=0)
	in_mean = in_mean.filled(in_mean.mean())

	# get completeness
	out_mean_comp = comp_edges.mean(axis=1)
	out_mean_comp = out_mean_comp.filled(out_mean_comp.mean())

	in_mean_comp = comp_edges.mean(axis=0)
	in_mean_comp = in_mean_comp.filled(in_mean_comp.mean())

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			edges_mean,
			out_mean,
			in_mean,
			out_mean_comp,
			in_mean_comp,
			variant_labels
		)),
		columns=['v measure', 'homo out', 'homo in', 'comp out', 'comp in', 'label'],
	)

	# sns.displot(df, x='v measure', hue='label', multiple="dodge", 
	# 			kde=True, stat='probability', common_norm=False, legend=True)
	# plt.show()
	# sns.displot(df, x='comp out', hue='label', multiple="dodge", 
	# 			kde=True, stat='probability', common_norm=False, legend=True)
	# plt.show()
	# sns.displot(df, x='comp in', hue='label', multiple="dodge", 
	# 			kde=True, stat='probability', common_norm=False, legend=True)
	# plt.show()

	# box plots
	sns.boxplot(
		x=(['min'] * len(variant_labels) 
			+ ['out homogeneity'] * len(variant_labels)
			+ ['in homogeneity'] * len(variant_labels)
			+ ['out completeness'] * len(variant_labels)
			+ ['in completeness'] * len(variant_labels)),
		y=np.hstack((edges_mean, out_mean, in_mean, out_mean_comp, in_mean_comp)), 
		hue=np.hstack((variant_labels, variant_labels, variant_labels, variant_labels, variant_labels))
	)
	plt.show()

	# plot precision recall and roc curves
	precision, recall, _ = precision_recall_curve(
		variant_labels, -edges_mean, pos_label=0)
	fpr, tpr, _ = roc_curve(variant_labels, -edges_mean, pos_label=0)

	print('AUC v measure: {}'.format(roc_auc_score(variant_labels, edges_mean)))

	out_precision, out_recall, _ = precision_recall_curve(
		variant_labels, -out_mean, pos_label=0)
	out_fpr, out_tpr, _ = roc_curve(variant_labels, -out_mean, pos_label=0)

	print('AUC homogeneity out: {}'.format(roc_auc_score(variant_labels, out_mean)))

	in_precision, in_recall, _ = precision_recall_curve(
		variant_labels, -in_mean, pos_label=0)
	in_fpr, in_tpr, _ = roc_curve(variant_labels, -in_mean, pos_label=0)

	print('AUC homogeneity in: {}'.format(roc_auc_score(variant_labels, in_mean)))

	out_precision_comp, out_recall_comp, _ = precision_recall_curve(
		variant_labels, -out_mean_comp, pos_label=0)
	out_fpr_comp, out_tpr_comp, _ = roc_curve(variant_labels, -out_mean_comp, pos_label=0)

	print('AUC completeness out: {}'.format(roc_auc_score(variant_labels, out_mean_comp)))

	in_precision_comp, in_recall_comp, _ = precision_recall_curve(
		variant_labels, -in_mean_comp, pos_label=0)
	in_fpr_comp, in_tpr_comp, _ = roc_curve(variant_labels, -in_mean, pos_label=0)

	print('AUC completeness in: {}'.format(roc_auc_score(variant_labels, in_mean_comp)))

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(recall, precision, label='min')
	plt.plot(out_recall, out_precision, label='out homogeneity')
	plt.plot(in_recall, in_precision, label='in homogeneity')
	plt.plot(out_recall_comp, out_precision_comp, label='out completeness')
	plt.plot(in_recall_comp, in_precision_comp, label='in completeness')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1.01)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(fpr, tpr, label='min')
	plt.plot(out_fpr, out_tpr, label='out homogeneity')
	plt.plot(in_fpr, in_tpr, label='in homogeneity')
	plt.plot(out_fpr_comp, out_tpr_comp, label='out completeness')
	plt.plot(in_fpr_comp, in_tpr_comp, label='in completeness')
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()