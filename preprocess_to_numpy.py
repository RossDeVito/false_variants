from utils import *


if __name__ == '__main__':
	fragments_path='data/fragments/chr20_1-5M/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-5M/1.0.potential_SNVs.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
	
	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	print('Original fragments:')
	matrix_sparsity_info(fragments, print_info=True)

	# get real/false variant labels
	variant_labels = get_true_variants(
		longshot_vcf_path=longshot_vcf_path,
		ground_truth_vcf_path=ground_truth_vcf_path
	)

	# save preprocessed data
	save_preprocessed(
		'data/preprocessed/chr20_1-5M.npz',
		fragments,
		qualities,
		variant_labels
	)

	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-5M.npz'
	)
	print('\nOriginal fragments reloaded as preprocessed:')
	matrix_sparsity_info(fragments, print_info=True)