"""The LUCB algorithm originally proposed in

Kalyanakrishnan, S., Tewari, A., Auer, P., & Stone, P. "PAC subset 
selection in stochastic multi-armed bandits." ICML 2012.

with "improved" LUCB++ version that results in better performance by

Simchowitz, M., Jamieson, K., Recht, B. "Towards a Richer Understanding of 
Adaptive Sampling in the Moderate-Confidence Regime." Preprint 2016.
"""
import time
import math
from numpy import sqrt, log, exp, mean, cumsum, zeros, argsort, argmin, argmax, array
import numpy
from sortedcontainers import SortedListWithKey
import ipdb
from confidence_bounds import ConfidenceBound

numpy.set_printoptions(precision=4)
numpy.set_printoptions(linewidth=200)

class LUCB(object):
	def __init__(self, n, k=1, delta=0.05, epsilon=0.0, bound_type='SubGaussian_LIL', improved=False, extra_rules = 0):
		self.name = 'LUCB'
		if improved: 
			self.name += '++' 
		self.n = n # no arms
		self.k = k
		self.delta = delta
		self.epsilon = epsilon
		self.bound_type = bound_type
		self.improved = improved
		self.extra_rules = extra_rules
		self.reset()

	def reset(self):
		self.total_queries = 0
		self.controlbest = True # means that if it doesn't stop before trunctime, returns true
		self._should_stop = False
		self.next_index = None
		# arms list contains the original indeces (of the mu_list)
		self.arms = []
		self.mu_hat_decreasing = SortedListWithKey(key = lambda x: -x['mu_hat'])
		self.ucb_decreasing = SortedListWithKey(key = lambda x: -x['ucb'])
		# Draw all of them uniformly for 10
		
		for i in range(self.n):
			# 'T' number of times the arm was pulled
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
		# Incorporates additional stopping rules
		TOP = self.mu_hat_decreasing[0:self.k]
		TOP = sorted(TOP, key = lambda y: y['lcb'])
		lcb_arm = TOP[0]
		if lcb_arm['lcb'] > stop_threshold:
			return True

		# If LCB > UCB then it is also returned
		return self._should_stop

	def get_query(self):
		self.total_queries += 1 # Starts with 1
		if self.total_queries % 2: # Pull both LCB and UCB arm
			TOP = self.mu_hat_decreasing[0:self.k]
			# sorts from small to big
			TOP = sorted(TOP, key = lambda y: y['lcb'])

			# Get arm with highest UCB
			best_arm = sorted(self.arms, key = lambda y: -y['mu_hat'])[0]
			maxucb_arm = sorted(self.arms, key = lambda y: -y['ucb'])[0]
			# Get control arm
			controlarm = self.arms[0]

			# Get top k arm with smallest LCB 
			lcb_arm = TOP[0]		
			self.mu_hat_decreasing.remove(lcb_arm)
			self.ucb_decreasing.remove(lcb_arm)
			
			# Get lower n-k arms with highest UCB
			tmp_idx = 0
			while self.ucb_decreasing[tmp_idx] in TOP:
				tmp_idx += 1
			ucb_arm = self.ucb_decreasing[tmp_idx]
			self.mu_hat_decreasing.remove(ucb_arm)
			self.ucb_decreasing.remove(ucb_arm)

			self.next_index = ucb_arm['index']

			#print ("best arm mu: %f and LCB: %f" % (lcb_arm['mu_hat'],lcb_arm['lcb']))
			#print ("seco arm mu: %f and UCB: %f" % (ucb_arm['mu_hat'], ucb_arm['ucb']))
			
			if self.extra_rules == 1:
				# Stopping rule different for epsilon > 0 to only get 
				# arms which are eps better than control
				# Does this stopping rule make sense? For eps = 0 it's the usual rule
				# For control stop if the eps modified "LCB of control" is > UCB all i ~= 0 
				if controlarm['lcb'] + self.epsilon > maxucb_arm['ucb']:
					self._should_stop = True
					self.controlbest = True	
				# Stop and declare bestarm if it's the best and at least eps better than control UCB
				elif (lcb_arm['lcb'] > ucb_arm['ucb']) & (lcb_arm['lcb'] > controlarm['ucb'] + self.epsilon):
					self._should_stop = True
					self.controlbest = False
				elif ( lcb_arm['ucb'] < ucb_arm['lcb'] + self.epsilon)  & (lcb_arm['lcb'] > controlarm['ucb'] + self.epsilon):
					# Stop if all arms are within \eps with each other,
					# i.e. best arm so far is smaller than smallest bla. 
					# Could use different confidence interval here
					self._should_stop = True
					self.controlbest = False
				
			elif lcb_arm['lcb'] > ucb_arm['ucb']:
				self._should_stop = True
				self.controlbest = False

			return lcb_arm['index']
		else:
			return self.next_index 

	def report_answer(self, idx, answer, verbose = 0):
		
		arm = self.arms[idx]
		arm['Xsum'] += answer
		arm['T'] += 1.
		arm['mu_hat'] = arm['Xsum']/arm['T']
		unionb = self.n
		cb = ConfidenceBound(self.bound_type)
		# if self.delta/unionb == 0:
		#  	ipdb.set_trace()
		if self.improved: unionb = 2.*(self.n-self.k)	
		arm['lcb'] = cb.lower(arm['mu_hat'], self.delta/float(unionb), arm['T'])
		if self.improved: unionb = 2.*self.k
		arm['ucb'] = cb.upper(arm['mu_hat'], self.delta/float(unionb), arm['T'])
		self.mu_hat_decreasing.add(arm)
		self.ucb_decreasing.add(arm)
		if verbose:
			print("Arm %d mu: %f LCB: %f UCB: %f" % (idx, arm['mu_hat'], arm['lcb'], arm['ucb']))

	def recommended_subset(self):
		return self.mu_hat_decreasing[0:self.k]

