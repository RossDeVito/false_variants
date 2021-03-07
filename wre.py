import numpy as np
from scipy import stats


def weighted_entropy(site1, site2):
	site1 = np.array(site1)
	site2 = np.array(site2)

	if len(site1) != len(site2):
		raise ValueError('site1 and site2 should be same length')

	# make probability tables
	where_s1_0 = (site1 == 0)
	global w 
	w = site1
	num_s1_0 = where_s1_0.sum()
	where_s1_1 = (site1 == 1)
	num_s1_1 = where_s1_1.sum()

	if num_s1_0 == 0:
		p_0_given_0 = .5
		p_1_given_0 = .5
	else:
		p_0_given_0 = np.count_nonzero(site2[where_s1_0] == 0) / num_s1_0
		p_1_given_0 = np.count_nonzero(site2[where_s1_0] == 1) / num_s1_0

	if num_s1_1 == 0:
		p_0_given_1 = .5
		p_1_given_1 = .5
	else:
		p_0_given_1 = np.count_nonzero(site2[where_s1_1] == 0) / num_s1_1
		p_1_given_1 = np.count_nonzero(site2[where_s1_1] == 1) / num_s1_1

	ent_given_0 = stats.entropy([p_0_given_0, p_1_given_0])
	ent_given_1 = stats.entropy([p_0_given_1, p_1_given_1])

	return (num_s1_0 * ent_given_0 + num_s1_1 * ent_given_1) / (num_s1_0 + num_s1_1)


def min_weighted_entropy(site1, site2):
	res = (weighted_entropy(site1, site2), weighted_entropy(site2, site1))
	print(res)
	return min(res)


if __name__ == '__main__':
	print(min_weighted_entropy([0,0,1,1], [0,0,1,1]))
	print(min_weighted_entropy([0,0,1,1], [1,1,0,0]))
	print(min_weighted_entropy([0,0,1,1], [1,1,1,1]))

	print(min_weighted_entropy([0,0,1,1,1,1,1,1,1,1,1], [1,0,1,1,1,1,1,1,1,1,1]))
	print(min_weighted_entropy([0,0,0,0,0,1,1,1,1,1,1], [1,0,1,1,1,1,1,1,1,1,1]))
	print(min_weighted_entropy([1,1,1,0,1,1,1,1,1,1,1], [1,0,1,1,1,1,1,1,1,1,1]))
	print(min_weighted_entropy([1,1,1,0,1,1,1,1,1,1,1], [0,0,1,0,0,0,0,0,0,0,1]))

	print(min_weighted_entropy([0,0,0,0,1,1,1,1], [1,1,1,0,0,1,1,1]))
	print(min_weighted_entropy([0,0,0,0,1,1,1,1], [1,1,1,0,0,0,0,0]))


	