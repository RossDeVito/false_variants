import warnings

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import roc_auc_score, classification_report

from utils import load_preprocessed, matrix_sparsity_info

'''
np.set_printoptions(edgeitems=5, linewidth=1000)
'''


def get_connections(site1, site2):
	''' return mask of rows which are fragments common to both sites '''
	return (~np.isnan(site1)) & (~np.isnan(site2))


def get_best_f1_ind(precision, recall):
	fscore = (2 * precision * recall) / (precision + recall)
	return np.argmax(fscore)


if __name__ == '__main__':
	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-1M.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	# get undirected edge weights
	min_connections = 2

	edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

	# to catch places where phi undefined and therefore should be no edge
	with warnings.catch_warnings():
		warnings.simplefilter("error")

		for i, j in tqdm(zip(*np.tril_indices(fragments.shape[1], k=-1)), 
						desc='edges',
						total=((fragments.shape[1] ** 2) / 2 - fragments.shape[1])):

			# check if two sites are connected by enough fragments
			connections = get_connections(fragments[:,i], fragments[:,j])

			if connections.sum() < min_connections:
				continue

			try:
				corr = np.abs(matthews_corrcoef(
					fragments[:,i][connections], 
					fragments[:,j][connections]
				))
			except Warning:
				pass
			else:
				edges[i,j] = corr
				edges[j,i] = corr

	# get average of edges
	edges_mean = edges.mean(axis=1)
	edges_mean = edges_mean.filled(edges_mean.mean())

	np.save('data/results/phi_correlation_1M.npy', edges_mean)

	reweighted_edges = edges_mean * edges

	rw_edges_mean = reweighted_edges.mean(axis=1)
	rw_edges_mean = rw_edges_mean.filled(rw_edges_mean.mean())

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			edges_mean,
			rw_edges_mean,
			variant_labels
		)),
		columns=['correlation', 'reweighted correlation', 'label'],
	)

	sns.displot(df, x='correlation', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()
	sns.displot(df, x='reweighted correlation', hue='label', multiple="dodge", 
				kde=True, stat='probability', common_norm=False, legend=True)
	plt.show()

	# box plots
	sns.boxplot(
		x=['mean correlation'] * len(variant_labels) + ['reweighted'] * len(variant_labels),
		y=np.hstack((edges_mean, rw_edges_mean)), 
		hue=np.hstack((variant_labels, variant_labels))
	)
	plt.show()

	# plot precision recall and roc curves
	precision, recall, threshold = precision_recall_curve(
		variant_labels, -edges_mean, pos_label=0)
	fpr, tpr, _ = roc_curve(variant_labels, -edges_mean, pos_label=0)
 
	phi_thresh = threshold[get_best_f1_ind(precision, recall)]
	pred_y = edges_mean > (-phi_thresh)

	print('AUC: {}'.format(roc_auc_score(variant_labels, edges_mean)))
	print(classification_report(variant_labels, pred_y, labels=[0,1],
                             	target_names=['false variant', 'true variant']))

	rw_precision, rw_recall, _ = precision_recall_curve(
		variant_labels, -rw_edges_mean, pos_label=0)
	rw_fpr, rw_tpr, _ = roc_curve(variant_labels, -rw_edges_mean, pos_label=0)

	print('AUC: {}'.format(roc_auc_score(variant_labels, rw_edges_mean)))

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(recall, precision, label='Phi correlation')
	plt.plot(rw_recall, rw_precision, label='reweighted correlation')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1.01)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(fpr, tpr, label='Phi correlation')
	plt.plot(rw_fpr, rw_tpr, label='reweighted correlation')
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()