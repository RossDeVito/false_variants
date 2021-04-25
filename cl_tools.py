import argparse
import re
import sys

from likelihood import run_cl_site_probs


class MyParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


def run_site_prob(args):
	''' Runs site_prob command line tool '''
	run_cl_site_probs(
		args.fragments_path, 
		args.vcf_path, 
		[args.prob_site_index] + args.other_site_inds,
		args.alpha,
		progress_bar=args.progress
	)


parser = MyParser()
subparsers = parser.add_subparsers()

site_prob_parser = subparsers.add_parser('site_prob')
site_prob_parser.add_argument(
	'fragments_path',
	help='Path to fragments.txt',
)
site_prob_parser.add_argument(
	'vcf_path',
	help='Path to Longshot output 2.0.realigned_genotypes.vcf',
)
site_prob_parser.add_argument(
	'alpha',
	type=float,
	help='alpha value to use in prior (typically 0.001-0.0015)',
)
site_prob_parser.add_argument(
	'prob_site_index',
	type=int,
	help='Index in fragments matrix of site to evaluate probabilities for'
)
site_prob_parser.add_argument(
	'other_site_inds',
	type=int, 
	nargs='*',
	help='Indices of other sites to use in evaluating probabilities'
)
site_prob_parser.add_argument(
	'-p', '--progress',
	default=True,
	help='Show pregress bar (True by default)',
	action="store_true"
)
usage = site_prob_parser.format_usage()[7:]  # remove "usage: " prefix
site_prob_parser.usage = re.sub(r'\[(.+?) \[\1 ...\]\]', r'[\1 ...]', usage)
site_prob_parser.set_defaults(func=run_site_prob)


if __name__ == '__main__':
	'''
	example bash:

	FRAGPATH='data/fragments/chr20_1-1M/fragments.txt'
	VCFPATH='data/fragments/chr20_1-1M/2.0.realigned_genotypes.vcf'

	python cl_tools.py site_prob $FRAGPATH $VCFPATH .001 10 12 13
	'''
	args = parser.parse_args()
	args.func(args)
