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
		'data/preprocessed/chr20_1-5M.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	# get undirected edge weights
	min_connections = 2

	edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	homo_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

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
		# 	out homogeneity is the same as in completeness
		#	in homogeneity is the same as out completeness
		homo_edges[i,j] = homo
		homo_edges[j,i] = comp

	# get average of edges
	edges_mean = edges.mean(axis=1)
	edges_mean = edges_mean.filled(edges_mean.mean())
 
	np.save('data/results/v_measure_5M.npy', edges_mean)

	# get homogeneity
	out_mean = homo_edges.mean(axis=1)
	out_mean = out_mean.filled(out_mean.mean())

	np.save('data/results/homogeneity_out_5M.npy', out_mean)

	in_mean = homo_edges.mean(axis=0)
	in_mean = in_mean.filled(in_mean.mean())
 
	np.save('data/results/homogeneity_in_5M.npy', in_mean)

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			edges_mean,
			out_mean,
			in_mean,
			variant_labels
		)),
		columns=['v measure', 'homo out', 'homo in', 'label'],
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
		x=(['v measure'] * len(variant_labels) 
			+ ['out homogeneity'] * len(variant_labels)
			+ ['in homogeneity'] * len(variant_labels)),
		y=np.hstack((edges_mean, out_mean, in_mean)), 
		hue=np.hstack((variant_labels, variant_labels, variant_labels))
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

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(recall, precision, label='min')
	plt.plot(out_recall, out_precision, label='out homogeneity')
	plt.plot(in_recall, in_precision, label='in homogeneity')
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
	plt.xlabel('false positive rate (positive label: false variant)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()