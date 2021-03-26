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
	print('Best threshold: ', np.abs(thresholds[np.argmax(f1_scores)]))
	print('Best F1-Score: ', np.max(f1_scores))

	return np.max(f1_scores), thresholds[np.argmax(f1_scores)]


if __name__ == '__main__':
	# load preprocessed data
	_, _, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)

	save_dir = 'data/results'
	
	res_saves = [
		('fisher_p05_5M.npy', 'fisher .05'),
		('fisher_p01_5M.npy', 'fisher .01'),
		('fisher_p005_5M.npy', 'fisher .005'),
		('chi_p05_5M.npy', 'chi .05'),
		('chi_p01_5M.npy', 'chi .01'),
		('chi_p005_5M.npy', 'chi .005'),
		('phi_correlation_5M.npy', 'phi correlation')
	]

	results = [(np.load(os.path.join(save_dir, s[0])), s[1]) for s in res_saves]

	for i, r in enumerate(results):
		print(r[1])

		# flip
		if r[1] == 'phi correlation':
			results[i] = (-r[0], r[1])

		get_best_f1(*precision_recall_curve(variant_labels, -results[i][0], pos_label=0))
		print()

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

	# get confusion matrices at different cuttoffs

	fig, axs = plt.subplots(2, 5)

	for i, r in enumerate(results):
		precision, recall, thresholds = precision_recall_curve(
			variant_labels, -r[0], pos_label=0
		)

		thresh = -thresholds[-1]

		preds = r[0] > thresh

		cm = confusion_matrix(variant_labels, preds)#, normalize='true')

		ax	= axs[i,0]
		sns.heatmap(cm, annot=True, ax = ax, fmt='g')
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Best Precision ({:.3})'.format(precision[-2])) 
		# ax.xaxis.set_ticklabels(['business', 'health'])
		# ax.yaxis.set_ticklabels(['health', 'business'])

		thresh_ind = np.where(precision > .98)[0].min()

		thresh = -thresholds[thresh_ind]

		preds = r[0] > thresh

		cm = confusion_matrix(variant_labels, preds)#, normalize='true')

		ax	= axs[i,1]
		sns.heatmap(cm, annot=True, ax = ax, fmt='g')
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Precision > .98') 
		# ax.xaxis.set_ticklabels(['business', 'health'])
		# ax.yaxis.set_ticklabels(['health', 'business'])

		thresh_ind = np.where(precision > .975)[0].min()

		thresh = -thresholds[thresh_ind]

		preds = r[0] > thresh

		cm = confusion_matrix(variant_labels, preds)#, normalize='true')

		ax	= axs[i,2]
		sns.heatmap(cm, annot=True, ax = ax, fmt='g')
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Precision > .975') 
		# ax.xaxis.set_ticklabels(['business', 'health'])
		# ax.yaxis.set_ticklabels(['health', 'business'])

		thresh_ind = np.where(precision > .95)[0].min()

		thresh = -thresholds[thresh_ind]

		preds = r[0] > thresh

		cm = confusion_matrix(variant_labels, preds)#, normalize='true')

		ax	= axs[i,3]
		sns.heatmap(cm, annot=True, ax = ax, fmt='g')
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Precision > .95') 
		# ax.xaxis.set_ticklabels(['business', 'health'])
		# ax.yaxis.set_ticklabels(['health', 'business'])

		best_f1, thresh = get_best_f1(precision, recall, thresholds)

		preds = r[0] > -thresh

		cm = confusion_matrix(variant_labels, preds)#, normalize='true')

		ax	= axs[i,4]
		sns.heatmap(cm, annot=True, ax = ax, fmt='g')
		ax.set_xlabel('Predicted labels')
		ax.set_ylabel('True labels')
		ax.set_title('Best f1 ({:.3})'.format(best_f1)) 
		# ax.xaxis.set_ticklabels(['business', 'health'])
		# ax.yaxis.set_ticklabels(['health', 'business'])

		break
