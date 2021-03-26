import numpy as np
import allel


def read_fragments(fragments_path):
	''' 
	Reads fragments file and returns read allele and quality data as 
		numpy matrices. Sites in matrices where there is no data are np.nan

	Args:	path to a fragments.txt file

	Returns 3 numpy arrays:
		array of fragment_ids corresponding to rows
		matrix of allele values {0,1,np.nan} where rows correspond to samples 
			and cols to sites
		matrix of Phred quality scores or np.nan where no read
	'''
	frag_ids, row_col_pairs, allele_vals, qual_scores = read_fragments_arrays(
		fragments_path
	)

	allele_mat = np.full(row_col_pairs.max(axis=0)+1, np.nan)
	allele_mat[row_col_pairs[:,0], row_col_pairs[:,1]] = allele_vals

	qual_mat = np.full(row_col_pairs.max(axis=0)+1, np.nan)
	qual_mat[row_col_pairs[:,0], row_col_pairs[:,1]] = qual_scores

	return frag_ids, allele_mat, qual_mat


def read_fragments_arrays(fragments_path):
	''' 
	Reads fragments file and returns data as numpy arrays of corresponding
		indices and data

	Args:	path to a fragments.txt file

	Returns 4 numpy arrays:
		fragment_ids corresponding to rows
		row col indices for the data in allele_vals and qual_scores
		allele values {0,1} at row,col locs
		Phred quality scores at row,col locs
	'''
	with open(fragments_path) as f:

		frag_ids = []		# fragment ids from col 2
		row_col_pairs = []	# row,col indices coresponding to allele values
		allele_vals = []	# 	and quality scores as matrices
		qual_scores = []

		row_ind = 0

		for line in f:
			line_data = line.strip().split()
			frag_ids.append(line_data[1])

			# get sample's row,col pairs and allele vals
			block_data = line_data[2:-1]

			for i in range(0, len(block_data), 2):
				block_start_ind = int(block_data[i])

				for start_offset in range(len(block_data[i + 1])):
					row_col_pairs.append(
						(row_ind, block_start_ind + start_offset)
					)
					allele_vals.append(block_data[i + 1][start_offset])

			# add quality scores
			qual_str = line_data[-1]
			for char in qual_str:
				qual_scores.append(ord(char) - 33)

			row_ind += 1

		# set indices to start at 0
		row_col_pairs = np.array(row_col_pairs)
		row_col_pairs -= row_col_pairs.min(axis=0, keepdims=True)

		return (
			np.array(frag_ids),
			row_col_pairs,
			np.array(allele_vals).astype(int),
			np.array(qual_scores)
		)


def get_bed_mask(bed_path, ls_callset_pos, chrom='chr20'):
	''' 
	Reads areas in GIAB from .bed and uses to mask longshot callset positions
	'''
	with open(bed_path) as f:
		starts = []
		ends = []

		for line in f:
			line_data = line.strip().split()

			if line_data[0] == chrom:
				starts.append(line_data[1])
				ends.append(line_data[2])

	starts = np.array(starts).astype(int)
	ends = np.array(ends).astype(int)

	in_bed_range = []

	for ls_pos in ls_callset_pos:
		in_bed_range.append(
			np.any((starts <= ls_pos) & (ends > ls_pos))
		)

	return np.array(in_bed_range)


def get_true_variants(longshot_vcf_path, ground_truth_vcf_path, giab_bed_path,
						return_vcfs=False):
	''' 
	Finds true/false variants in fragments file using GIAB ground truth vcf

	Args: 
		longshot_vcf_path: path to "4.0.final_genotypes.vcf" for longshot run
			that produced the fragments.txt file being used
		ground_truth_vcf_path: path to GIAB ground truth vcf
		giab_bed_path: giab ground truth corresponding .bed
		return_vcfs: if the longshot and ground_truth vcfs should be returned.
			Will be returned as callsets
	
	Returns:
		array length of number of cols of fragments matrix where each is
			labeled True/False wrt being real variants
		site_mask to use to filter columns of fragments matrix
		if return_vcfs, returns (true_variants, longshot_vcf, ground_truth_vcf)
	'''
	# load vcf from longshot run
	ls_callset = allel.read_vcf(longshot_vcf_path)

	# load ground truth vcf
	callset = allel.read_vcf(ground_truth_vcf_path)

	# find true variants
	chr20_mask = callset['variants/CHROM'] == ls_callset['variants/CHROM'][0]
	callset = mask_callset(callset, chr20_mask)

	in_truth = np.in1d(ls_callset['variants/POS'], callset['variants/POS'])

	# mask out regions not in .bed
	in_bed_mask = get_bed_mask(giab_bed_path, ls_callset['variants/POS'])

	# find where longshot predicts heterozygous
	ls_01 = np.all(np.equal(ls_callset['calldata/GT'], [0,1]), axis=2).T[0]
	ls_10 = np.all(np.equal(ls_callset['calldata/GT'], [1,0]), axis=2).T[0]
	ls_hetero = ls_01 | ls_10

	site_mask = in_bed_mask & ls_hetero

	if return_vcfs:
		return in_truth.astype(int)[site_mask], site_mask, ls_callset, callset
	else:
		return in_truth.astype(int)[site_mask], site_mask


def mask_callset(callset, mask):
	for key in list(callset):
		if key == 'samples':
			continue
		callset[key] = callset[key][mask]

	return callset


def matrix_sparsity_info(allele_mat, print_info=False):
	''' 
	Get info about sparsity of allele/quality/incorrect read matrix by
	rows and cols
	'''

	sample_reads = np.count_nonzero(~np.isnan(allele_mat), axis=1)
	site_reads = np.count_nonzero(~np.isnan(allele_mat), axis=0)

	if print_info:
		nonzero_sites = np.count_nonzero(~np.isnan(allele_mat))

		print("num elements not missing:\t{}".format(nonzero_sites))
		print("percent matrix not missing:\t{:.3f}".format(
			nonzero_sites / allele_mat.size
		))

		print("num fragments:\t{}".format(allele_mat.shape[0]))
		print("num sites:\t{}".format(allele_mat.shape[1]))

		print("\nfragments:")
		val = np.mean(sample_reads)
		print("\tmean reads:\t{:.1f}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.median(sample_reads)
		print("\tmedian reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.max(sample_reads)
		print("\tmax reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		val = np.min(sample_reads)
		print("\tmin reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[1]))
		
		print("\nsites:")
		val = np.mean(site_reads)
		print("\tmean reads:\t{:.1f}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.median(site_reads)
		print("\tmedian reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.max(site_reads)
		print("\tmax reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))
		val = np.min(site_reads)
		print("\tmin reads:\t{}\t{:.3f}".format(val, val / allele_mat.shape[0]))

	return sample_reads, site_reads


def save_preprocessed(path, fragments, qualities, variant_labels):
	np.savez(
		path, 
		fragments=fragments,
		qualities=qualities,
		variant_labels=variant_labels)


def load_preprocessed(path):
	''' Returns (fragments, qualities, variant_labels) from .npz '''
	data = np.load(path)
	return data['fragments'], data['qualities'], data['variant_labels']


if __name__ == '__main__':
	fragments_path='data/fragments/chr20_1-500K/fragments.txt'
	longshot_vcf_path='data/fragments/chr20_1-500K/2.0.realigned_genotypes.vcf'
	ground_truth_vcf_path='data/GIAB/HG002_GRCh38_1_22_v4.1_draft_benchmark.vcf'
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
		'data/preprocessed/chr20_1-500K.npz',
		fragments,
		qualities,
		variant_labels
	)

	# load preprocessed data
	fragments, qualities, variant_labels = load_preprocessed(
		'data/preprocessed/chr20_1-500K.npz'
	)
	print('New fragments reloaded as preprocessed:')
	matrix_sparsity_info(fragments, print_info=True)

# plt.subplots()
# plt.subplot(1,2,1)
# sns.histplot(site_reads, ax=plt.gca(), kde=True)
# plt.title("Reads per Site")
# plt.xlabel("Num. Reads")
# plt.subplot(1,2,2)
# sns.histplot(sample_reads, ax=plt.gca(), kde=True)
# plt.title("Reads per Fragment")
# plt.xlabel("Num. Reads")
# plt.show()