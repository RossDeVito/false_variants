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

	fisher_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
	chi2_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
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
		contingency_table = pd.crosstab(
			fragments[:,j][connections], 
			fragments[:,i][connections]
		)

		if contingency_table.shape != (2,2):
			continue

		fisher_p_val = stats.fisher_exact(contingency_table)[1]

		fisher_edges[i,j] = fisher_p_val
		fisher_edges[j,i] = fisher_p_val

		chi2_p_val = stats.chi2_contingency(contingency_table)[1]

		chi2_edges[i,j] = chi2_p_val
		chi2_edges[j,i] = chi2_p_val


	# get average of edges at different p-values
	fp_1 = (fisher_edges < .1).mean(axis=1)
	fp_1 = fp_1.filled(fp_1.mean())

	fp_05 = (fisher_edges < .05).mean(axis=1)
	fp_05 = fp_05.filled(fp_05.mean())

	fp_01 = (fisher_edges < .01).mean(axis=1)
	fp_01 = fp_01.filled(fp_01.mean())

	chi_p_1 = (chi2_edges < .1).mean(axis=1)
	chi_p_1 = chi_p_1.filled(chi_p_1.mean())

	chi_p_05 = (chi2_edges < .05).mean(axis=1)
	chi_p_05 = chi_p_05.filled(chi_p_05.mean())

	chi_p_01 = (chi2_edges < .01).mean(axis=1)
	chi_p_01 = chi_p_01.filled(chi_p_01.mean())

	# load phi correlations on 5 mil
	phi = -np.load('data/results/phi_correlation_5M.npy')
	_, _, phi_variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)

	# plot distributions
	df = pd.DataFrame(
		np.column_stack((
			fp_1,
			fp_05,
			fp_01,
			chi_p_1,
			chi_p_05,
			chi_p_01,
			variant_labels
		)),
		columns=[
			'fisher (p-val < .1)', 'fisher (p-val < .05)', 'fisher (p-val < .01)',
			'chi^2 (p-val < .1)', 'chi^2 (p-val < .05)', 'chi^2 (p-val < .01)',
			'label'
		],
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
		x=(['fisher (p-val < .1)'] * len(variant_labels) 
			+ ['fisher (p-val < .05)'] * len(variant_labels)
			+ ['fisher (p-val < .01)'] * len(variant_labels)
			+ ['chi^2 (p-val < .1)'] * len(variant_labels)
			+ ['chi^2 (p-val < .05)'] * len(variant_labels)
			+ ['chi^2 (p-val < .01)'] * len(variant_labels)
			+ ['phi'] * len(phi_variant_labels)),
		y=np.hstack((fp_1, fp_05, fp_01, chi_p_1, chi_p_05, chi_p_01, phi)), 
		hue=np.hstack((variant_labels, variant_labels, variant_labels, 
						variant_labels, variant_labels, variant_labels,
						phi_variant_labels))
	)
	plt.show()

	# plot precision recall and roc curves
	phi_precision, phi_recall, _ = precision_recall_curve(
		phi_variant_labels, -phi, pos_label=0)
	phi_fpr, phi_tpr, _ = roc_curve(phi_variant_labels, -phi, pos_label=0)

	print('phi: {}'.format(roc_auc_score(phi_variant_labels, phi)))

	fp1_precision, fp1_recall, _ = precision_recall_curve(
		variant_labels, -fp_1, pos_label=0)
	fp1_fpr, fp1_tpr, _ = roc_curve(variant_labels, -fp_1, pos_label=0)

	print('Fisher (p-val < .1): {}'.format(roc_auc_score(variant_labels, fp_1)))

	fp05_precision, fp05_recall, _ = precision_recall_curve(
		variant_labels, -fp_05, pos_label=0)
	fp05_fpr, fp05_tpr, _ = roc_curve(variant_labels, -fp_05, pos_label=0)

	print('Fisher (p-val < .05): {}'.format(roc_auc_score(variant_labels, fp_05)))

	fp01_precision, fp01_recall, _ = precision_recall_curve(
		variant_labels, -fp_01, pos_label=0)
	fp01_fpr, fp01_tpr, _ = roc_curve(variant_labels, -fp_01, pos_label=0)

	print('Fisher (p-val < .01): {}'.format(roc_auc_score(variant_labels, fp_01)))

	cp1_precision, cp1_recall, _ = precision_recall_curve(
		variant_labels, -chi_p_1, pos_label=0)
	cp1_fpr, cp1_tpr, _ = roc_curve(variant_labels, -chi_p_1, pos_label=0)

	print('Chi^2 (p-val < .1): {}'.format(roc_auc_score(variant_labels, chi_p_1)))

	cp05_precision, cp05_recall, _ = precision_recall_curve(
		variant_labels, -chi_p_05, pos_label=0)
	cp05_fpr, cp05_tpr, _ = roc_curve(variant_labels, -chi_p_05, pos_label=0)

	print('Chi^2 (p-val < .05): {}'.format(roc_auc_score(variant_labels, chi_p_05)))

	cp01_precision, cp01_recall, _ = precision_recall_curve(
		variant_labels, -chi_p_01, pos_label=0)
	cp01_fpr, cp01_tpr, _ = roc_curve(variant_labels, -chi_p_01, pos_label=0)

	print('Chi^2 (p-val < .01): {}'.format(roc_auc_score(variant_labels, chi_p_01)))

	plt.subplots()
	plt.subplot(1,2,1)
	plt.plot(phi_recall, phi_precision, label='Phi')
	plt.plot(fp1_recall, fp1_precision, label='Fisher (p-val < .1)')
	plt.plot(fp05_recall, fp05_precision, label='Fisher (p-val < .05)')
	plt.plot(fp01_recall, fp01_precision, label='Fisher (p-val < .01)')
	plt.plot(cp1_recall, cp1_precision, label='Chi^2 (p-val < .1)')
	plt.plot(cp05_recall, cp05_precision, label='Chi^2 (p-val < .05)')
	plt.plot(cp01_recall, cp01_precision, label='Chi^2 (p-val < .01)')
	plt.xlabel('false variant recall')
	plt.xlim(0, 1.01)
	plt.ylabel('false variant precision')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.subplot(1,2,2)
	plt.plot(phi_fpr, phi_tpr, label='Phi')
	plt.plot(fp1_fpr, fp1_tpr, label='Fisher (p-val < .1)')
	plt.plot(fp05_fpr, fp05_tpr, label='Fisher (p-val < .05)')
	plt.plot(fp01_fpr, fp01_tpr, label='Fisher (p-val < .01)')
	plt.plot(cp1_fpr, cp1_tpr, label='Chi^2 (p-val < .1)')
	plt.plot(cp05_fpr, cp05_tpr, label='Chi^2 (p-val < .05)')
	plt.plot(cp01_fpr, cp01_tpr, label='Chi^2 (p-val < .01)')
	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)

	plt.show()