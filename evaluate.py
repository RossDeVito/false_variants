from functools import reduce

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_curve, roc_curve


def plot_curves(variant_labels, curves=[], points=[], pos_label=1, paired=False):
	if paired:
		sns.set_palette('Paired')

	plt.subplots()
	plt.subplot(1,2,1)
	minor_ticks = np.arange(0, 1, .05)
	major_ticks = np.arange(0, 1.01, .1)

	for pred, desc in points:
		prec = precision_score(variant_labels, pred, pos_label=pos_label)
		recall = recall_score(variant_labels, pred, pos_label=pos_label)

		plt.plot(recall, prec, '.', label=desc)

	if pos_label == 1:
		for vals, desc in curves:
			precision, recall, threshold = precision_recall_curve(
				variant_labels, vals, pos_label=pos_label)

			plt.plot(recall, precision, label=desc)

		plt.xlabel('false variant recall')
		plt.ylabel('false variant precision')


	elif pos_label == 0:
		for vals, desc in curves:
			precision, recall, threshold = precision_recall_curve(
				variant_labels, -vals, pos_label=pos_label)

			plt.plot(recall, precision, label=desc)

		plt.xlabel('true variant recall')
		plt.ylabel('true variant precision')

	plt.xlim(0, 1.01)
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)
	plt.gca().set_xticks(major_ticks)
	plt.gca().set_yticks(major_ticks)
	plt.gca().set_xticks(minor_ticks, minor=True)
	plt.gca().set_yticks(minor_ticks, minor=True)
	plt.grid(True, which='minor', alpha=.3)

	plt.subplot(1,2,2)

	# so colors match
	for p in points:
		next(plt.gca()._get_lines.prop_cycler)

	if pos_label == 1:
		for vals, desc in curves:
			fpr, tpr, _ = roc_curve(variant_labels, vals, pos_label=pos_label)
			
			plt.plot(fpr, tpr, label=desc)

		plt.xlabel('false positive rate (positive label: false variant)')

	elif pos_label == 0:
		for vals, desc in curves:
			fpr, tpr, _ = roc_curve(variant_labels, -vals, pos_label=pos_label)
			
			plt.plot(fpr, tpr, label=desc)

		plt.xlabel('false positive rate (positive label: true variant)')

	plt.xlim(0, 1.01)
	plt.ylabel('true positive rate')
	plt.ylim(0, 1.01)
	plt.legend(loc='lower right')
	plt.title('ROC Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True, which='major')
	plt.gca().set_xticks(major_ticks)
	plt.gca().set_yticks(major_ticks)
	plt.gca().set_xticks(minor_ticks, minor=True)
	plt.gca().set_yticks(minor_ticks, minor=True)
	plt.grid(True, which='minor', alpha=.3)


def add_predictions(df):
	''' 
	Adds prediction info columns and reduces to just sites in .bed and
	with a genotype of 0, 1, or 2.
	'''
	# reduce to prediction sites
	df = df.loc[df.in_bed == 1]
	df = df.loc[df.genotype != -1]
	df['genotype'] = df.genotype.astype(int)

	# add cols for predictions
	df['pred_genotype'] = np.argmax(df.iloc[:, -3:].values, axis=1)
	df['pred_homozygous'] = (df.pred_genotype != 1).astype(int)
	df['log_ods'] = np.log( (df.P0 + df.P2) / df.P1 )

	return df

if __name__ == '__main__':
	append_and_save = True
	save_path = 'data/results/1M_predicitions_joined.tsv'

	# paired=True
	# files_to_eval = [
	# 	'data/results/1M_predicitions_w1_a001.tsv',
	# 	'data/results/1M_predicitions_w1_a0015.tsv',
	# 	'data/results/1M_predicitions_w2_a001.tsv',
	# 	'data/results/1M_predicitions_w2_a0015.tsv',
	# 	'data/results/1M_predicitions_w3_a001.tsv',
	# 	'data/results/1M_predicitions_w3_a0015.tsv',
	# 	'data/results/1M_predicitions_w4_a001.tsv',
	# ]
	paired = False
	# files_to_eval = [
	# 	'data/results/1M_predicitions_w1_a001.tsv',
	# 	'data/results/1M_predicitions_w2_a001.tsv',
	# 	'data/results/1M_predicitions_w3_a001.tsv',
	# 	'data/results/1M_predicitions_w4_a001.tsv',
	# 	'data/results/1M_predicitions_w5_a001.tsv',
	# ]
	files_to_eval = [
		'data/results/1M_predicitions_w1_a001.tsv',
		'data/results/1M_predicitions_w1_a0015.tsv',
		'data/results/1M_predicitions_w2_a001.tsv',
		'data/results/1M_predicitions_w2_a0015.tsv',
		'data/results/1M_predicitions_w3_a001.tsv',
		'data/results/1M_predicitions_w3_a0015.tsv',
		'data/results/1M_predicitions_w4_a001.tsv',
		'data/results/1M_predicitions_w5_a001.tsv',
	]

	results = []
	original_dfs = []

	for f in files_to_eval:
		orig_df = pd.read_csv(f, sep='\t')
		df = add_predictions(orig_df)

		if len(results) == 0:
			true_homo = (df.genotype != 1).astype(int)

		window_size = int(df.window_size.iloc[0])
		alpha = df.alpha.iloc[0]

		original_dfs.append((window_size, alpha, orig_df))
		results.append((window_size, alpha, df))

	# print('Genotype Prediction')
	# print(classification_report(df.genotype, df.pred_genotype))
	# print(confusion_matrix(df.genotype, df.pred_genotype))

	# print('Homozygosity Prediction')
	# true_homo = (df.genotype != 1).astype(int)
	# print(classification_report(true_homo, df.pred_homozygous))
	# print(confusion_matrix(true_homo, df.pred_homozygous))

	# Plot curves
	bin_formated_res = [(df.log_ods, 'w={} a={}'.format(w,a)) for w,a,df in results]
	plot_curves(true_homo, bin_formated_res, pos_label=1, paired=paired)
	plt.show()

	# Plot zoomed in PR curve
	plt.figure()
	pos_label=1
	lw=2

	minor_ticks = np.arange(0, 1, .05)
	major_ticks = np.arange(0, 1.01, .1)

	for vals, desc in bin_formated_res:
		precision, recall, threshold = precision_recall_curve(
			true_homo, vals, pos_label=pos_label)

		plt.plot(recall, precision, label=desc, lw=lw)

	plt.xlabel('false variant recall')
	plt.ylabel('false variant precision')

	plt.xlim(0, 1.01)
	plt.ylim(0, 1.01)
	plt.legend()
	plt.title('Precision Recall Curve')
	plt.gca().set_aspect('equal', adjustable='box')
	plt.grid(True)
	plt.gca().set_xticks(major_ticks)
	plt.gca().set_yticks(major_ticks)
	plt.gca().set_xticks(minor_ticks, minor=True)
	plt.gca().set_yticks(minor_ticks, minor=True)
	plt.grid(True, which='minor', alpha=.3)

	plt.show()

	# Plot genotype confusion matrices
	geno_cm_res = [(df, 'w={} a={}'.format(w,a)) for w,a,df in results]

	fig, axs = plt.subplots(1, len(results))

	for i, r in enumerate(geno_cm_res):
		ax = axs[i]
		ax.set_title('{}'.format(r[1]))
		
		preds = r[0].pred_genotype
		cm = confusion_matrix(r[0].genotype, preds, normalize=None)	
			
		sns.heatmap(cm, annot=True, ax=ax, fmt='g', cbar=False,
			xticklabels=[0,1,2])
		ax.set_xlabel('Predicted genotype')
		if i == 0:
			ax.set_ylabel('True genotype')
		
		# ax.xaxis.set_ticklabels([0,1,2])
		# ax.yaxis.set_ticklabels(['hetero', 'homo'])
		ax.set_aspect("equal")

	fig.suptitle("Genotype Confusion Matrices")
	plt.show()

	# Plot binary confusion matrices
	fig, axs = plt.subplots(1, len(results))

	for i, r in enumerate(bin_formated_res):
		ax = axs[i]
		ax.set_title('{}'.format(r[1]))
		
		preds = (r[0] >= 0).astype(int)
		cm = confusion_matrix(true_homo, preds, normalize=None)	
			
		sns.heatmap(cm, annot=True, ax=ax, fmt='g', cbar=False)
		ax.set_xlabel('Predicted genotype')
		if i == 0:
			ax.set_ylabel('True genotype')
		
		ax.xaxis.set_ticklabels(['hetero', 'homo'])
		ax.yaxis.set_ticklabels(['hetero', 'homo'])
		ax.set_aspect("equal")

	fig.suptitle("Hetero/Homozygous Confusion Matrices")
	plt.show()

	if append_and_save:
		relabeled_dfs = []

		for n,(w,a,df) in enumerate(original_dfs):
			df.columns = [
				str(col) + '_{}'.format(n) if i > 6 else col for i,col in enumerate(df.columns)
			]
			relabeled_dfs.append(df)

		df_joined = reduce(
			lambda left,right: pd.merge(left,right,
					on=['site_ind','in_bed','chrom','pos','ref','alt','genotype']
				),
			relabeled_dfs
		)

		df_joined.to_csv(save_path, na_rep='', sep='\t', index=False)
		
