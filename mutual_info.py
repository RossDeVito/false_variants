import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import stats

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import rand_score, adjusted_rand_score
from sklearn.metrics import mutual_info_score, adjusted_mutual_info_score
from sklearn.metrics import normalized_mutual_info_score

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

	rand_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	adj_rand_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	mi_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	ami_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	nmi_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

	for i, j in tqdm(zip(*np.tril_indices(fragments.shape[1], k=-1)), 
					desc='edges',
					total=((fragments.shape[1] ** 2) / 2 - fragments.shape[1])):

		# check if two sites are connected by enough fragments
		connections = get_connections(fragments[:,i], fragments[:,j])

		if connections.sum() < min_connections:
			continue

		# rand index
		rand = rand_score(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)
		rand_edges[i,j] = rand
		rand_edges[j,i] = rand

		adj_rand = adjusted_rand_score(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)
		adj_rand_edges[i,j] = adj_rand
		adj_rand_edges[j,i] = adj_rand

		# mutual info
		mi = mutual_info_score(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)
		mi_edges[i,j] = mi
		mi_edges[j,i] = mi

		ami = adjusted_mutual_info_score(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)
		ami_edges[i,j] = ami
		ami_edges[j,i] = ami

		nmi = normalized_mutual_info_score(
			fragments[:,i][connections], 
			fragments[:,j][connections]
		)
		nmi_edges[i,j] = nmi
		nmi_edges[j,i] = nmi

	# get average of edges
	rand_mean = rand_edges.mean(axis=1)
	rand_mean = rand_mean.filled(rand_mean.mean())

	adj_rand_mean = adj_rand_edges.mean(axis=1)
	adj_rand_mean = adj_rand_mean.filled(adj_rand_mean.mean())

	mi_mean = mi_edges.mean(axis=1)
	mi_mean = mi_mean.filled(mi_mean.mean())

	ami_mean = ami_edges.mean(axis=1)
	ami_mean = ami_mean.filled(ami_mean.mean())

	nmi_mean = nmi_edges.mean(axis=1)
	nmi_mean = nmi_mean.filled(nmi_mean.mean())

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			rand_mean,
			adj_rand_mean,
			mi_mean,
			ami_mean,
			nmi_mean,
			variant_labels
		)),
		columns=['rand', 'adj_rand', 'mutual info', 'ami', 'nmi', 'label'],
	)

	sns.displot(df, x='rand', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='adj_rand', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='mutual info', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='ami', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='nmi', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()

	# box plots
	sns.boxplot(
		x=(['rand'] * len(variant_labels) 
			+ ['adj rand'] * len(variant_labels)
			+ ['mutual info'] * len(variant_labels)
			+ ['ami'] * len(variant_labels)
			+ ['nmi'] * len(variant_labels)
			),
		y=np.hstack((rand_mean, adj_rand_mean, mi_mean, ami_mean, nmi_mean)), 
		hue=np.hstack((variant_labels, variant_labels, variant_labels,
						variant_labels,variant_labels))
	)
	plt.show()

	# plot precision recall and roc curves
	rand_precision, rand_recall, _ = precision_recall_curve(
		variant_labels, -rand_mean, pos_label=0)
	rand_fpr, rand_tpr, _ = roc_curve(variant_labels, -rand_mean, pos_label=0)

	print('AUC rand: {}'.format(roc_auc_score(variant_labels, rand_mean)))

	adj_rand_precision, adj_rand_recall, _ = precision_recall_curve(
		variant_labels, -adj_rand_mean, pos_label=0)
	adj_rand_fpr, adj_rand_tpr, _ = roc_curve(variant_labels, -adj_rand_mean, pos_label=0)

	print('AUC adj rand: {}'.format(roc_auc_score(variant_labels, adj_rand_mean)))

	mi_precision, mi_recall, _ = precision_recall_curve(
		variant_labels, -mi_mean, pos_label=0)
	mi_fpr, mi_tpr, _ = roc_curve(variant_labels, -mi_mean, pos_label=0)

	print('AUC mi: {}'.format(roc_auc_score(variant_labels, mi_mean)))

	ami_precision, ami_recall, _ = precision_recall_curve(
		variant_labels, -ami_mean, pos_label=0)
	ami_fpr, ami_tpr, _ = roc_curve(variant_labels, -ami_mean, pos_label=0)

	print('AUC ami: {}'.format(roc_auc_score(variant_labels, ami_mean)))

	nmi_precision, nmi_recall, _ = precision_recall_curve(
		variant_labels, -nmi_mean, pos_label=0)
	nmi_fpr, nmi_tpr, _ = roc_curve(variant_labels, -nmi_mean, pos_label=0)

	print('AUC nmi: {}'.format(roc_auc_score(variant_labels, nmi_mean)))

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(rand_recall, rand_precision, label='rand')
	plt.plot(adj_rand_recall, adj_rand_precision, label='adj rand')
	plt.plot(mi_recall, mi_precision, label='mutual info')
	plt.plot(ami_recall, ami_precision, label='adjusted mutual info')
	plt.plot(nmi_recall, nmi_precision, label='normalized mutual info')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1.01)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(rand_fpr, rand_tpr, label='rand')
	plt.plot(adj_rand_fpr, adj_rand_tpr, label='adj rand')
	plt.plot(mi_fpr, mi_tpr, label='mutual info')
	plt.plot(ami_fpr, ami_tpr, label='adjusted mutual info')
	plt.plot(nmi_fpr, nmi_tpr, label='normalized mutual info')
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()