"""A number of confidence bounds for different statistical 
settings (Gaussians and Bernoullis) as a function of time."""

from scipy.stats import norm as normal
from numpy import sqrt, log, exp, cumsum
from numpy.random import randn


def _binary_search(a,b,f):
	"""f(x) returns True if threshold is greater than x, False otherwise."""
	precision = 1e-6
	pivot = (a+b)/2.
	while b-a > precision:
		pivot = (a+b)/2.
		if f(pivot):
			b = pivot
		else:
			a = pivot
	return pivot

class ConfidenceBound(object):
	"""Provides class for confidence bound for different statistical settings.

	Intended use:
		mu_hat = mean(mu + randn() for _ in range(t))

		cb = ConfidenceBound('SubGaussian')
		u = cb.upper(mu_hat, delta, t, anytime=True)
		l = cb.lower(mu_hat, delta, t, anytime=True)

		Then l <= mu <= u with probability at least 1-delta. If anytime=True then
		this relation holds for all t simultaneously. 
	"""

	def __init__(self, name='SubGaussian', anytime=True):
		"""Parameters of confidence bound determined by name.
		All subGaussian tail-bounds are for scale-1 RVs.

		Args:
			anytime: Boolean inidcating that the bound should hold for all time simultaneously.
				If not LIL bound, anytime bound is accomplished by summable 1/(2*t^2) union bound.
			name: what kind of RV
				SubGaussian: typical sub-gaussian tail bound of scale 1
				Gaussian_Exact: tail probability for Gaussian of scale 1
				SubGaussian_LIL: SubGaussian tail bound but using LIL union bound instead of 
					the trivial union bound set by setting anytime=True with SubGaussian bound.
					When using this bound name, anytime must be set equal to True or raises exception
				Bernoulli_LIL: rewards Bernoulli, uses SubGaussian-LIL bound 
				Bernoulli: rewards are bernoulli in {0, 1} and uses Chernoff KL bound
		"""
		self.name = name
		self.anytime = anytime

		if 'LIL' in self.name:
			assert self.anytime==True, 'When using \'SubGaussian_LIL\' you must set anytime==True'

	def _kl(self, a, b):
		if a == 0:
			return -log(1.-b)
		elif a == 1:
			return -log(b)
		elif b == 0:
			return float('inf')
		elif b == 1:
			return float('inf')
		else:
			return a*log(a/b) + (1.-a)*log((1.-a)/(1.-b))

	def lower(self, mu_hat, delta, t=1):
		if delta==0 :
			ipdb.set_trace()
		if self.name in ['Bernoulli_LIL', 'SubGaussian_LIL', 'SubGaussian', 'Gaussian_Exact']:
			return 2*mu_hat-self.upper(mu_hat, delta, t)
		elif self.name == 'Bernoulli':
			conf = delta
			if self.anytime: 
				conf = delta/(2.*t**2)
			f = lambda x: t*self._kl(mu_hat, x) < log(1./conf)
			return _binary_search(0., mu_hat, f)

	# CURRENTLY ONLY SUPPORTING Gaussians with sigma = 1!
	def upper(self, mu_hat, delta, t=1, sigma = 1):
		if 'Bernoulli' in self.name:
			sigma = 0.5 # SubGaussian parameter
		if 'LIL' in self.name:
			#if self.name =='SubGaussian_LIL': (same for Bernoulli LIL?)
			beta = log(1./delta) + 3*log(log(1./delta)) + 1.5*log(log(exp(1)*t))
			return mu_hat + sqrt(2.*pow(sigma,2)*beta/t)
		else:
			conf = delta
			if self.anytime: 
				conf = delta/(2.*t**2)

			if self.name == 'SubGaussian':
				return mu_hat + sqrt(2.*log(1./conf)/t)
			elif self.name == 'Gaussian_Exact':
				f = lambda x: 1-normal.cdf(x) < conf
				return mu_hat + _binary_search(0., sqrt(2.*log(1./conf)), f)/sqrt(t)
			elif self.name == 'Bernoulli':
				f = lambda x: t*self._kl(mu_hat, x) > log(1./conf)
				return _binary_search(mu_hat, 1., f)



def plot_cb(N=1000, delta=0.05):
	times = range(1,N+1)

	bound_type = 'SubGaussian'
	cb = ConfidenceBound(bound_type)
	upper = [cb.upper(0, delta, t)*t for t in times]
	plt.plot(times, upper, label=bound_type+'-log(t)')

	bound_type = 'SubGaussian_LIL'
	cb = ConfidenceBound(bound_type)
	upper = [cb.upper(0, delta, t)*t for t in times]
	plt.plot(times, upper, label=bound_type)

	bound_type = 'Gaussian_Exact'
	cb = ConfidenceBound(bound_type)
	upper = [cb.upper(0, delta, t)*t for t in times]
	plt.plot(times, upper, label=bound_type+'-log(t)')

	bound_type = 'SubGaussian'
	cb = ConfidenceBound(bound_type, anytime=False)
	upper = [cb.upper(0, delta, t)*t for t in times]
	plt.plot(times, upper, '--', label=bound_type+'-1')

	bound_type = 'Gaussian_Exact'
	cb = ConfidenceBound(bound_type, anytime=False)
	upper = [cb.upper(0, delta, t)*t for t in times]
	plt.plot(times, upper, '--', label=bound_type+'-1')

	plt.legend()
	plt.title('Upper Confidence Bounds')
	plt.xlabel('Number of samples')
	plt.show()

def test_cb_validity(N=100000, delta=0.05):
	# Multiplier for delta = 0.05:
	#		Gaussian_Exact = 1.7
	#		SubGaussian = 11.6
	#		SubGaussian_LIL = 56
	times = range(1,N+1)

	bound_type = 'SubGaussian_LIL'
	cb = ConfidenceBound(bound_type)
	upper = [cb.upper(0, delta, t)*t for t in times]
	
	cnt = 0
	max_reps = 1000000
	for rep in range(max_reps):
		Xsum = cumsum(randn(N))
		for i in range(N):
			if Xsum[i] > upper[i]:
				cnt += 1
				break
		if rep % 1000 == 0 and rep > 0:
			delta_hat = float(cnt)/float(rep)
			if delta_hat > 0:
				ratio = delta/delta_hat
			else:
				ratio = float('NaN')
			print (rep, delta_hat, ratio)


if __name__ == "__main__":
	
	plot_cb(N=100, delta=0.05)

	test_cb_validity(N=100, delta=0.05)




