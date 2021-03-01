import numpy as np
from sklearn.metrics import classification_report

from utils import *


if __name__ == '__main__':
	fragments_path='data/fragments/chr20_1-500K/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-500K/1.0.potential_SNVs.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	
	# load read fragments and the quality of each read location
	_, fragments, qualities = read_fragments(fragments_path)

	print('Fragments:')
	matrix_sparsity_info(fragments, print_info=True)

	# get real/false variant labels
	variant_labels = get_true_variants(
		longshot_vcf_path=longshot_vcf_path,
		ground_truth_vcf_path=ground_truth_vcf_path
	)

	# given fragments (and possibly qualities) label each col as a real (1)
	#	or false (0) variant. Simple baselines are shown below:

	# predict all varients are false
	pred_labels = np.zeros_like(variant_labels)
	cr = classification_report(variant_labels, pred_labels, zero_division=0)
	print('\nPredict all false:\n' + cr)

	# predict random
	rand_labels = np.random.choice((0,1), size=variant_labels.shape)
	cr = classification_report(variant_labels, rand_labels)
	print('\nPredict randomly:\n' + cr)
