import warnings
from multiprocessing import Pool

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


def mp_run_tests(args):
	i = args[0]
	j = args[1]
	var1 = args[2]
	var2 = args[3]

	# check if two sites are connected by enough fragments
	connections = get_connections(var1, var2)

	if connections.sum() < min_connections:
		return []

	# undirected edges
	contingency_table = pd.crosstab(
		var1[connections], 
		var2[connections]
	)

	if contingency_table.shape != (2,2):
		return []

	fisher_p_val = stats.fisher_exact(contingency_table)[1]
	chi2_p_val = stats.chi2_contingency(contingency_table)[1]

	return [
		('fisher', i, j, fisher_p_val), ('chi', i, j, chi2_p_val)
	]


def to_test_gen(matrix_dim):
	for i, j in zip(*np.tril_indices(matrix_dim, k=-1)):
		yield (i, j, fragments[:,i], fragments[:,j])

if __name__ == '__main__':
	load_matrices = True
	save_matrices = False

	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)
	matrix_sparsity_info(fragments, print_info=True)

	if load_matrices:
		fisher_edges = np.ma.masked_invalid(
			np.load('data/results/fisher_mat_5M.npy')
		)
		chi2_edges = np.ma.masked_invalid(
			np.load('data/results/chi_mat_5M.npy')
		)
	else:
		# get undirected edge weights
		min_connections = 2

		fisher_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
		chi2_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))
		comp_edges = np.ma.masked_all((fragments.shape[1], fragments.shape[1]))

		with Pool(8) as p:
			r = list(tqdm(
					p.imap(mp_run_tests, to_test_gen(fragments.shape[1])),
					total=int((fragments.shape[1] ** 2) / 2 - fragments.shape[1]), 
					desc='running tests'
				))

		for res in tqdm(r, desc='filling matrices'):
			for p in res:
				if p[0] == 'fisher':
					fisher_edges[p[1], p[2]] = p[3]
					fisher_edges[p[2], p[1]] = p[3]
				elif p[0] == 'chi':
					chi2_edges[p[1], p[2]] = p[3]
					chi2_edges[p[2], p[1]] = p[3]
				else:
					raise ValueError('invalid test type')

	if save_matrices:
		print("Saving p-value matrices")
		np.save('data/results/fisher_mat_5M.npy', fisher_edges.filled(np.nan))
		np.save('data/results/chi_mat_5M.npy', chi2_edges.filled(np.nan))


	# get average of edges at different p-values
	fp_1 = (fisher_edges < .1).mean(axis=1)
	fp_1 = fp_1.filled(fp_1.mean())
	np.save('data/results/fisher_p1_5M.npy', fp_1)

	fp_05 = (fisher_edges < .05).mean(axis=1)
	fp_05 = fp_05.filled(fp_05.mean())
	np.save('data/results/fisher_p05_5M.npy', fp_05)

	fp_01 = (fisher_edges < .01).mean(axis=1)
	fp_01 = fp_01.filled(fp_01.mean())
	np.save('data/results/fisher_p01_5M.npy', fp_01)

	chi_p_1 = (chi2_edges < .1).mean(axis=1)
	chi_p_1 = chi_p_1.filled(chi_p_1.mean())
	np.save('data/results/chi_p1_5M.npy', chi_p_1)

	chi_p_05 = (chi2_edges < .05).mean(axis=1)
	chi_p_05 = chi_p_05.filled(chi_p_05.mean())
	np.save('data/results/chi_p05_5M.npy', chi_p_05)

	chi_p_01 = (chi2_edges < .01).mean(axis=1)
	chi_p_01 = chi_p_01.filled(chi_p_01.mean())
	np.save('data/results/chi_p01_5M.npy', chi_p_01)

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