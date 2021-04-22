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


def plot_curves(variant_labels, curves=[], points=[], pos_label=0):
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
			if 'entropy' in desc or 'misread' in desc:
				precision, recall, threshold = precision_recall_curve(
					variant_labels, -vals, pos_label=pos_label)
			else:
				precision, recall, threshold = precision_recall_curve(
					variant_labels, vals, pos_label=pos_label)

			plt.plot(recall, precision, label=desc)

		plt.xlabel('true variant recall')
		plt.ylabel('true variant precision')

	elif pos_label == 0:
		for vals, desc in curves:
			if 'entropy' in desc or 'misread' in desc:
				precision, recall, threshold = precision_recall_curve(
					variant_labels, vals, pos_label=pos_label)
			else:
				precision, recall, threshold = precision_recall_curve(
					variant_labels, -vals, pos_label=pos_label)

			plt.plot(recall, precision, label=desc)

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

	plt.subplot(1,2,2)

	# so colors match
	for p in points:
		next(plt.gca()._get_lines.prop_cycler)

	if pos_label == 1:
		for vals, desc in curves:
			if 'entropy' in desc or 'misread' in desc:
				fpr, tpr, _ = roc_curve(variant_labels, -vals, pos_label=pos_label)
			else:
				fpr, tpr, _ = roc_curve(variant_labels, vals, pos_label=pos_label)
			
			plt.plot(fpr, tpr, label=desc)

		plt.xlabel('false positive rate (positive label: true variant)')

	elif pos_label == 0:
		for vals, desc in curves:
			if 'entropy' in desc or 'misread' in desc:
				fpr, tpr, _ = roc_curve(variant_labels, vals, pos_label=pos_label)
			else:
				fpr, tpr, _ = roc_curve(variant_labels, -vals, pos_label=pos_label)
			
			plt.plot(fpr, tpr, label=desc)

		plt.xlabel('false positive rate (positive label: false variant)')

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
		print(preds.sum())

		return confusion_matrix(variant_labels, preds, normalize=normalize), None

if __name__ == '__main__':
	# load preprocessed data
	_, _, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-1M.npz'
	)

	save_dir = 'data/results'
	
	# original
	res_saves = [
		('phi_correlation_1M.npy', 'phi correlation'),
		('likelihood_w1_1M.npy', 'likelihood w=1'),
		('likelihood_w2_1M.npy', 'likelihood w=2'),
		('likelihood_w3_1M.npy', 'likelihood w=3'),
		('likelihood_w4_1M.npy', 'likelihood w=4'),
		('likelihood_w5_1M.npy', 'likelihood w=5'),
	]

	results = [(np.load(os.path.join(save_dir, s[0])), s[1]) for s in res_saves]

	for i, r in enumerate(results):
		print(r[1])
		print(
			get_best_f1(*precision_recall_curve(variant_labels, -results[i][0], pos_label=0))
		)

	# curves
	plot_curves(variant_labels, results, pos_label=0)
	plt.show()

	# confusion matrices
	fig, axs = plt.subplots(1, len(results))

	for i, r in enumerate(results):
		ax = axs[i]
		if 'likelihood' in r[1]:
			ax.set_title('{}'.format(r[1]))
			preds = (r[0] >= 1).astype(int)
			cm = confusion_matrix(variant_labels, preds, normalize=None)
		else:
			ax.set_title('{} (best f1)'.format(r[1]))
			cm, best_f1 = get_cm_at_cutoff('f1', r[0], variant_labels)		
			
		sns.heatmap(cm, annot=True, ax=ax, fmt='g', cbar=False)
		ax.set_xlabel('Predicted genotype')
		if i == 0:
			ax.set_ylabel('True genotype')
		
		ax.xaxis.set_ticklabels(['homo', 'hetero'])
		ax.yaxis.set_ticklabels(['homo', 'hetero'])
		ax.set_aspect("equal")

	fig.suptitle("Confusion Matrices")
	plt.show()