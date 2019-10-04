import os

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

def save_plot(x, obj, title):
	plt.figure()
	plt.plot(x,obj)
	plt.savefig(os.path.join('_fig',title+'.png'))


class DescreteBayesianInterence():
	def __init__(self, kind):
		self.kind = kind
		if self.kind == 'bernoulli':
			self.a, self.b = 2, 2
			self.prior = stats.beta
			self.observation = stats.bernoulli

		elif self.kind == 'category':
			self.alpha = np.random.rand(3)
			self.prior = stats.dirichlet
			self.observation = stats.multinomial

		elif self.kind == 'poisson':
			self.a, self.b = 2, 0.5
			self.prior = stats.gamma
			self.observation = stats.poisson

		else:
			print('{0} does not exists'.format(self.kind))
			exit()

	def fit(self, X):
		if self.kind == 'bernoulli':
			N = X.shape[0]
			hat_a = np.sum(X) + self.a
			hat_b = N - np.sum(X) + self.b
			return hat_a, hat_b

		elif self.kind == 'category':
			hat_alpha = np.sum(X, axis=0) + self.alpha
			return hat_alpha

		else:
			N = X.shape[0]
			hat_a = np.sum(X) + self.a
			hat_b = N + self.b
			return hat_a, hat_b

class ContinuousBayesianInference():
	def __init__(self):
		self.m, self.beta, self.a, self.b = 0.5, 1, 1, 1
		self.prior_gauss = stats.norm
		self.prior_gamma = stats.gamma
		self.observation = stats.norm

	def fit(self, X):
		N = X.shape[0]
		hat_beta = N + self.beta
		hat_m = (np.sum(X)+self.beta*self.m)/hat_beta
		hat_a = N/2 + self.a
		hat_b = (np.sum(X**2)+self.beta*(self.m**2)-hat_beta*(hat_m**2))/2 + self.b

		return hat_beta, hat_m, hat_a, hat_b



def main():
	'''
	# 3.2.1
	dbi = DescreteBayesianInterence(kind='bernoulli')
	x = np.linspace(-0.5,1.5,1000)
	save_plot(x, dbi.prior(dbi.a,dbi.b).pdf(x), 'prior')
	X = stats.bernoulli.rvs(p=0.8,size=100)
	hat_a, hat_b = dbi.fit(X)
	save_plot(x, dbi.prior(hat_a,hat_b).pdf(x), 'posterior')
	'''
	'''
	# 3.2.2
	dbi = DescreteBayesianInterence(kind='category')
	p_init = [0.3,0.5,0.2]
	X = stats.multinomial(n=1,p=p_init).rvs(size=100)
	hat_alpha = dbi.fit(X)
	print('setting:{0}, initial:{1}, inference:{2}'.format(p_init,dbi.prior(dbi.alpha).rvs().tolist(),dbi.prior(hat_alpha).rvs().tolist()))
	'''
	'''
	# 3.2.3
	dbi = DescreteBayesianInterence(kind='poisson')
	x = np.linspace(0,10,1000)
	save_plot(x,dbi.prior(dbi.a,scale=1./dbi.b).pdf(x),'prior_gamma')
	X = stats.poisson(mu=7).rvs(size=100)
	hat_a, hat_b = dbi.fit(X)
	save_plot(x,dbi.prior(hat_a,scale=1./hat_b).pdf(x),'posterior_gamma')
	'''

	# 3.3.3
	cbi = ContinuousBayesianInference()

	x = np.linspace(-1,1,1000)
	l = cbi.prior_gamma(cbi.a,cbi.b).rvs(size=1)[0]
	mu = cbi.prior_gauss(cbi.m,1/(cbi.beta*l)).rvs(size=1)[0]
	save_plot(x,cbi.observation(mu,1/l).pdf(x),'prior_gauss')

	X = stats.norm().rvs(size=100)
	hat_beta, hat_m, hat_a, hat_b = cbi.fit(X)

	# precision parameter increasing means uncertainty decreasing
	l_inference = cbi.prior_gamma(hat_a,hat_b).rvs(size=1)[0]
	mu_inference = cbi.prior_gauss(hat_m,1/(hat_beta*l_inference)).rvs(size=1)[0]
	print('mu:{0}, presicion:{1}'.format(mu_inference,l_inference))
	save_plot(x,cbi.observation(mu_inference,1/l_inference).pdf(x),'posterior_gauss')

	

if __name__ == '__main__':
	main()