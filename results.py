import os

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import roc_auc_score, f1_score

from utils import *


def load_files(file_name_pairs, result_path):
	ret = []
	
	for file_name, method_name in file_name_pairs:
		ret.append(
			(np.load(os.path.join(result_path, file_name)), method_name)
		)
		
	return ret


def get_best_f1_ind(precision, recall):
	fscore = (2 * precision * recall) / (precision + recall)
	return np.argmax(fscore)


if __name__ == '__main__':
	add_random = False
	results_path = 'data/results'

	points_saved = [
		('graph_5M.npy', 'estimated misread ratio max-cut')
	]

	curves_saved = [
		('phi_correlation_5M.npy', 'phi correlation'),
		('weighted_entropy_max_5M.npy', 'max edge weighted entropy'),
		('weighted_entropy_out_5M.npy', 'out edge weighted entropy'),
		('weighted_entropy_in_5M.npy', 'in edge weighted entropy'),
		('v_measure_5M.npy', 'V-measure'),
		('homogeneity_out_5M.npy', 'out edge homogeneity'),
		('homogeneity_in_5M.npy', 'in edge homogeneity'),
		('rand_index_5M.npy', 'Rand index'),
		('adj_rand_index_5M.npy', 'adjusted Rand index'),
		('mutual_info_5M.npy', 'mutual information'),
		('adj_mutual_info_5M.npy', 'adjusted mutual information'),
		('norm_mutual_info_5M.npy', 'normalized mutual information')
	]

	# entropies
	curves_saved = [
		('phi_correlation_5M.npy', 'phi correlation'),
		('weighted_entropy_max_5M.npy', 'max edge weighted entropy'),
		('weighted_entropy_out_5M.npy', 'out edge weighted entropy'),
		('weighted_entropy_in_5M.npy', 'in edge weighted entropy'),
	]

	# curves_saved = [
	# 	('phi_correlation_5M.npy', 'phi correlation'),
	# 	('weighted_entropy_out_5M.npy', 'out edge weighted entropy'),
	# 	('homogeneity_out_5M.npy', 'out edge homogeneity'),
	# 	('adj_rand_index_5M.npy', 'adjusted Rand index'),
	# 	('adj_mutual_info_5M.npy', 'adjusted mutual information'),
	# 	('norm_mutual_info_5M.npy', 'normalized mutual information'),
	# 	('emr_5M.npy', 'estimated misread ratio')
	# ]

	# for box plot
	curves_saved = [
		('phi_correlation_5M.npy', 'phi correlation'),
		('homogeneity_out_5M.npy', 'out edge homogeneity'),
		('adj_rand_index_5M.npy', 'adj. Rand index'),
		('adj_mutual_info_5M.npy', 'adj. mutual information (MI)'),
		('norm_mutual_info_5M.npy', 'normalized MI'),
		('weighted_entropy_out_5M.npy', 'out edge weighted entropy'),
		('emr_5M.npy', 'estimated misread ratio')
	]

	_, _, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)

	points = load_files(points_saved, results_path)
	curves = load_files(curves_saved, results_path)

	if add_random:
		curves.append(
			(np.random.random(curves[0][0].size), 'random')
		)

	# plot PR curve
	pos_label = 0

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
	
	plt.show()

	# dict of perfmance metrics at best f1
	curve_preds = []
	scores = dict()
	auc = []

	for vals, desc in curves:
		if 'entropy' in desc or 'misread' in desc:
			precision, recall, threshold = precision_recall_curve(
				variant_labels, vals, pos_label=pos_label)

			auc.append(roc_auc_score(variant_labels, -vals))

			thresh = threshold[get_best_f1_ind(precision, recall)]
			curve_preds.append(((vals < thresh).astype(int), desc))

		else:
			precision, recall, threshold = precision_recall_curve(
				variant_labels, -vals, pos_label=pos_label)

			auc.append(roc_auc_score(variant_labels, vals))

			thresh = threshold[get_best_f1_ind(precision, recall)]
			curve_preds.append(((vals > (-thresh)).astype(int), desc))
	
	auc.extend(0 for p in points)
	scores['AUC'] = auc
	curve_preds.extend(points)

	scores['false variant precision'] = []
	scores['false variant recall'] = []
	scores['false variant f1'] = []
	scores['macro precision'] = []
	scores['macro recall'] = []
	scores['macro f1'] = []

	for pred, desc in curve_preds:
		scores['false variant precision'].append(
			precision_score(variant_labels, pred, pos_label=0)
		)
		scores['false variant recall'].append(
			recall_score(variant_labels, pred, pos_label=0)
		)
		scores['false variant f1'].append(
			f1_score(variant_labels, pred, pos_label=0)
		)
		scores['macro precision'].append(
			precision_score(variant_labels, pred, average='macro')
		)
		scores['macro recall'].append(
			recall_score(variant_labels, pred, average='macro')
		)
		scores['macro f1'].append(
			f1_score(variant_labels, pred, average='macro')
		)

	results = pd.DataFrame(scores, index=[cp[1] for cp in curve_preds])
	print(results)

	# box plots
	xs = []
	ys = []
	hues = []

	for vals, desc in curves:
		xs.extend([desc] * len(vals))
		ys.extend(vals)
		hues.extend(variant_labels)

	with sns.plotting_context(font_scale=1.35):
		sns.boxplot(x=xs, y=ys, hue=hues)
		l = plt.legend()
		l.get_texts()[0].set_text('false variants')
		l.get_texts()[1].set_text('true variants')
		plt.show()