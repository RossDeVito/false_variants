from utils import *


if __name__ == '__main__':
	fragments_path='data/fragments/chr20_1-5M/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-5M/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf.gz'
	giab_bed_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.bed'
	
	# load read fragments and their qualities
	_, fragments, qualities = read_fragments(fragments_path)

	print('Original fragments:')
	matrix_sparsity_info(fragments, print_info=True)

	# get real/false variant labels
	variant_labels, site_mask, ls_callset, callset = get_true_variants(
		longshot_vcf_path=longshot_vcf_path,
		ground_truth_vcf_path=ground_truth_vcf_path,
		giab_bed_path=giab_bed_path,
		return_vcfs=True
	)

	# mask these with site_mask
	fragments = fragments[:, site_mask]
	qualities = qualities[:, site_mask]

	print('New fragments:')
	matrix_sparsity_info(fragments, print_info=True)

	# save preprocessed data
	save_preprocessed(
		'data/preprocessed/chr20_1-5M.npz',
		fragments,
		qualities,
		variant_labels
	)