"""UniformSampling allocates measurements uniformly to each arm.
Algorithm stops when upper and lower confidence bounds do not intersect
using trivial union bound over the n arms.  
"""
import time
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array
import numpy
from sortedcontainers import SortedListWithKey

from confidence_bounds import ConfidenceBound

numpy.set_printoptions(precision=4)
numpy.set_printoptions(linewidth=200)

class Uniform(object):
	
	def __init__(self, n, k=1, delta=0.05, epsilon=0.0, bound_type='SubGaussian_LIL'):
		self.name = 'Uniform'
		self.n = n
		self.k = k
		self.delta = delta
		self.epsilon = epsilon
		self.bound_type = bound_type
		self.improved = 0
		self.controlbest = False
		self.reset()

	def reset(self):
		self.total_queries = 0
		self._should_stop = False
		self.permutation = None

		self.arms = []
		self.mu_hat_decreasing = SortedListWithKey(key = lambda x: -x['mu_hat'])
		self.ucb_decreasing = SortedListWithKey(key = lambda x: -x['ucb'])
		for i in range(self.n):
			arm = {'index': i, 'Xsum': 0., 'T': 0, 'mu_hat': 0., 'ucb': float('inf'), 'lcb': -float('inf')}
			self.arms.append(arm)
			self.mu_hat_decreasing.add(arm)
			self.ucb_decreasing.add(arm)

	def should_stop(self, stop_threshold=float('inf')):
		"""Boolean indicating whether sampling should stop or not.

		Args:
			stop_threshold: if there exist k arms which can confidently be said 
				to have means greater than stop_threshold, method returns True
		"""
		TOP = self.mu_hat_decreasing[0:self.k]
		TOP = sorted(TOP, key = lambda y: y['lcb'])
		lcb_arm = TOP[0]
		if lcb_arm['lcb'] > stop_threshold:
			return True

		return self._should_stop

	def get_query(self):
		if self.total_queries % self.n == 0:
			TOP = self.mu_hat_decreasing[0:self.k]
			TOP = sorted(TOP, key = lambda y: y['lcb'])

			lcb_arm = TOP[0]

			tmp_idx = 0
			while self.ucb_decreasing[tmp_idx] in TOP:
				tmp_idx += 1
			ucb_arm = self.ucb_decreasing[tmp_idx]

			if lcb_arm['lcb'] > ucb_arm['ucb']:
				self._should_stop = True
				self.controlbest = False
				

		idx = self.total_queries % self.n
		ret_arm = self.arms[idx]
		self.mu_hat_decreasing.remove(ret_arm)
		self.ucb_decreasing.remove(ret_arm)

		self.total_queries += 1

		return idx #ret_arm['index']

	def report_answer(self, idx, answer, verbose = 0):
		arm = self.arms[idx]
		arm['Xsum'] += answer
		arm['T'] += 1.
		arm['mu_hat'] = arm['Xsum']/arm['T']
		unionb = self.n

		cb = ConfidenceBound(self.bound_type)
		if self.improved: unionb = 2.*(self.n-self.k)			
		arm['lcb'] = cb.lower(arm['mu_hat'], self.delta/unionb, arm['T'])
		if self.improved: unionb = 2.*self.k
		arm['ucb'] = cb.upper(arm['mu_hat'], self.delta/unionb, arm['T'])
		self.mu_hat_decreasing.add(arm)
		self.ucb_decreasing.add(arm)

		if verbose:
			print("Arm %d mu: %f LCB: %f UCB: %f" % (idx, arm['mu_hat'], arm['lcb'], arm['ucb']))


	def recommended_subset(self):
		return self.mu_hat_decreasing[0:self.k]

