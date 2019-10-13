import os, time, collections

import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

PLT = True

def log_sum_exp(X):
	# \log(\sum_{i=1}^{N}\exp(x_i))
	max_x = np.max(X, axis=1).reshape(-1,1)
	return np.log(np.sum(np.exp(X-max_x),axis=1).reshape(-1,1)) + max_x

# gibbs sampling (4.2)
def mixture_poisson_gibbs_sampling(X, K, max_iter):
	lmd = np.zeros((K,1)) + 1
	pi = np.zeros((K,1)) + 1/K
	a, b = 1, 1
	alpha = np.zeros((K,1)) + 1
	N = X.shape[0]

	sampled_lmd, sampled_pi, sampled_S = [], [], []
	for i in range(max_iter):
		# sample s_n
		tmp = X.dot(np.log(lmd).reshape(1,-1)) - lmd.reshape(1,-1) + np.log(pi).reshape(1,-1)
		# normalize eta
		log_Z = - log_sum_exp(tmp)
		eta = np.exp(tmp + log_Z)
		S = np.zeros((N, K))
		for n in range(N):
			S[n] = stats.multinomial.rvs(n=1,p=eta[n],size=1)
		sampled_S.append(S.copy())

		# sample lmd_k
		hat_a = X.T.dot(S).reshape(-1,1) + a
		hat_b = np.sum(S, axis=0).reshape(-1,1) + b
		for k in range(K):
			lmd[k] = stats.gamma.rvs(a=hat_a[k],scale=1/hat_b[k])
		sampled_lmd.append(lmd.copy())

		# sample pi
		hat_alpha = np.sum(S, axis=0).reshape(-1,1) + alpha
		pi = stats.dirichlet.rvs(hat_alpha.reshape(-1), size=1).reshape(-1,1)
		sampled_pi.append(pi.copy())

	return np.array(sampled_lmd).reshape(-1,K), np.array(sampled_pi).reshape(-1,K), np.array(sampled_S).reshape(-1,N,K)

def mixture_poisson_variational_inference(X, K, max_iter):
	init_a = np.ones((K,1))
	init_b = np.ones((K,1))
	init_alpha = np.random.rand(K,1)

	a = init_a.copy()
	b = init_b.copy()
	alpha = init_alpha.copy()

	for i in range(max_iter):
		# update q(s_n)
		ln_lmd_mean = special.digamma(a) - np.log(b)
		lmd_mean = a / b
		ln_pi_mean = special.digamma(alpha) - special.digamma(np.sum(alpha))
		tmp = X.dot(ln_lmd_mean.reshape(1,-1)) - lmd_mean.reshape(1,-1) + ln_pi_mean.reshape(1,-1)
		log_Z = - log_sum_exp(tmp)
		eta = np.exp(tmp + log_Z)

		# update q(lmd_k)
		a = X.T.dot(eta).reshape(-1,1) + init_a
		b = np.sum(eta, axis=0).reshape(-1,1) + init_b

		# update q(pi)
		alpha = np.sum(eta, axis=0).reshape(-1,1) + init_alpha

	return a, b, eta, alpha

def mixture_gaussian_gibbs_sampling(X, K, max_iter):
	'''
	X.shape = (n_sample,n_dim)
	'''
	N, D = X.shape
	# initialize mu, precision, pi
	m = np.zeros((D,1))
	beta = 0.1
	W = np.eye(D)
	nu = D + 1.0
	alpha = 100.0 * np.ones((K,1))
	
	pi = stats.dirichlet(alpha.reshape(-1)).rvs().reshape(-1,1)
	# precision = np.array([np.linalg.inv(K*np.eye(D)) for k in range(K)])
	precision = stats.wishart(df=nu,scale=W).rvs(size=3)
	# mu = np.random.uniform(X.min(), X.max(), (K, D))
	mu = np.array([stats.multivariate_normal(mean=m.reshape(-1),cov=precision[k]).rvs() for k in range(K)])

	sampled_mu, sampled_precision, sampled_S, sampled_pi = [], [], [], []
	for i in range(max_iter):

		# sample s_n
		tmp = [[(-(X[n]-mu[k]).T.dot(precision[k]).dot(X[n]-mu[k])/2 + np.log(np.linalg.det(precision[k]))/2 + np.log(pi[k]))[0] for k in range(K)] for n in range(N)]
		# normalize eta
		log_Z = - log_sum_exp(np.array(tmp))
		eta = np.exp(tmp + log_Z)
	
		S = np.zeros((N, K))
		for n in range(N):
			S[n] = stats.multinomial.rvs(n=1,p=eta[n],size=1)
		# print(collections.Counter(np.argmax(S,axis=1)).items())
		sampled_S.append(S.copy())

		# sample mu
		hat_beta = np.sum(S,axis=0) + beta
		hat_m = np.array([((np.sum([S[n,k]*X[n] for n in range(N)],axis=0))/hat_beta[k]) for k in range(K)])
		for k in range(K):
			mu[k] = stats.multivariate_normal(mean=hat_m[k],cov=np.linalg.inv(hat_beta[k]*precision[k])).rvs()
		sampled_mu.append(mu.copy())

		# sample precision
		hat_W = np.linalg.inv(np.array([np.dot(np.dot(X.T,np.diag(S[:,k])),X)- beta*np.dot(m,m.T) + hat_beta[k]*np.dot(hat_m[k].reshape(-1,1),hat_m[k].reshape(1,-1)) + np.linalg.inv(W) for k in range(K)]))
		hat_nu = np.sum(S,axis=0) + nu
		for k in range(K):
			precision[k] = stats.wishart(df=hat_nu[k],scale=hat_W[k]).rvs()
		sampled_precision.append(precision.copy())

		# sampled pi
		hat_alpha = np.sum(S, axis=0).reshape(-1,1) + alpha
		pi = stats.dirichlet.rvs(hat_alpha.reshape(-1), size=1).reshape(-1,1)
		sampled_pi.append(pi.copy())

	return np.array(sampled_mu), np.array(sampled_precision), np.array(sampled_S), np.array(sampled_pi)

def mixture_gaussian_variational_inference(X, K, max_iter):
	N, D = X.shape
	# initialize mu, precision, pi
	init_m = np.array([np.zeros((D,1)) for k in range(K)])
	init_beta = np.array([0.1 for k in range(K)])
	init_W = np.array([np.eye(D) for k in range(K)])
	init_nu = np.array([D + 1.0 for k in range(K)])
	init_alpha = 100.0 * np.ones((K,1))

	m, beta, W, nu, alpha = init_m.copy(), init_beta.copy(), init_W.copy(), init_nu.copy(), init_alpha.copy()

	for i in range(max_iter):
		# update q(s_n)
		# compute expection
		exp_precision = np.array([nu[k]*W[k] for k in range(K)])
		exp_ln_precision = np.array([(np.sum([special.digamma((nu[k]+d)/2) for d in range(D)]) + D*np.log(2) + np.log(np.linalg.det(W[k]))).reshape(-1) for k in range(K)])
		exp_precision_mu = np.array([nu[k]*np.dot(W[k],m[k]).reshape(-1) for k in range(K)])
		exp_muT_precision_mu = np.array([nu[k]*np.dot(np.dot(m[k].T,W[k]),m[k]).reshape(-1) + D/beta[k] for k in range(K)])
		exp_ln_pi = np.array([special.gamma(alpha[k]) - special.digamma(np.sum(alpha)) for k in range(K)])
		# print(exp_precision.shape,exp_ln_precision.shape,exp_precision_mu.shape,exp_muT_precision_mu.shape,exp_ln_pi.shape)
		
		tmp = [[(-X[n].T.dot(exp_precision[k]).dot(X[n])/2 + X[n].T.dot(exp_precision_mu[k]) - exp_muT_precision_mu[k]/2 + exp_ln_precision[k]/2 + exp_ln_pi[k])[0] for k in range(K)] for n in range(N)]
		print(tmp)
		# all values diverge to infinity
		log_Z = - log_sum_exp(np.array(tmp))
		eta = np.exp(tmp + log_Z)
		print(eta)
		exit()

		# update q(mu, precision)
		beta = np.array([np.sum(eta[:,k]) + beta[k] for k in range(K)])
		m = np.array([(np.sum([eta[n,k]*X[n] for n in range(N)],axis=0) + init_beta[k]*init_m[k].reshape(-1))/beta[k] for k in range(K)])
		# hat_W = np.linalg.inv(np.array([np.sum([eta[n,k]*np.dot(X[n].reshape(-1,1),X[n].reshape(1,-1)) for n in range(N)], axis=0) + beta*np.dot(m,m.T) - hat_beta[k]*np.dot(hat_m[k],hat_m[k].T) + np.linalg.inv(W) for k in range(K)]))
		W = []
		for k in range(K):
			W.append(np.linalg.inv(np.array(np.sum([eta[n,k]*np.dot(X[n].reshape(-1,1),X[n].reshape(1,-1)) for n in range(N)], axis=0) + init_beta[k]*np.dot(init_m[k],init_m[k].T) - beta[k]*np.dot(m[k],m[k].T) + np.linalg.inv(init_W[k]))))
		W = np.array(W)
		nu = np.array([np.sum(eta[:,k]) + init_nu[k] for k in range(K)])

		# update q(pi)
		alpha = np.array([np.sum(eta[:,k]) + init_alpha[k] for k in range(K)])

	return beta, m, W, nu, alpha, eta


def main():
	'''
	X = np.hstack((stats.poisson(15).rvs(size=300),stats.poisson(30).rvs(size=200))).reshape(-1,1)

	if PLT:
		plt.figure()
		plt.hist(X,bins=40)
		plt.show()
	'''
	'''
	lmd, pi, S = mixture_poisson_gibbs_sampling(X, 2, 100)
	if PLT:
		for i in range(2):
			plt.figure()
			plt.hist(lmd[:,i],bins=50)
			plt.show()
	'''
	'''
	a, b, eta, alpha = mixture_poisson_variational_inference(X, 2, 100)
	if PLT:
		x = np.linspace(0,50,1000)
		plt.figure()
		for i in range(2):
			plt.subplot(1,2,i+1)
			plt.plot(x,stats.gamma.pdf(x,a=a[i],scale=1/b[i]))
			plt.title('posterior lmd_'+str(i+1))
			plt.xlabel('lmd')
		plt.show()

		plt.figure()
		plt.hist(stats.dirichlet(alpha.reshape(-1)).rvs(size=1000)[:,0])
		plt.show()
	'''
	# '''
	X1 = stats.multivariate_normal(mean=np.array([0,5])).rvs(size=100)
	X2 = stats.multivariate_normal(mean=np.array([-6,1])).rvs(size=100)
	X3 = stats.multivariate_normal(mean=np.array([10,-1])).rvs(size=100)
	X = np.vstack((X1,X2,X3))

	if PLT:
		plt.scatter(X[:,0],X[:,1])
		plt.show()

	K = 3
	'''
	mu, precision, S, pi = mixture_gaussian_gibbs_sampling(X, K, 100)
	for k in range(K):
		print(collections.Counter(np.argmax(S[-1,k*100:(k+1)*100],axis=1)).items())
	print(mu[-1])
	'''
	hat_beta, hat_m, hat_W, hat_nu, hat_alpha, eta = mixture_gaussian_variational_inference(X, K ,100)

if __name__ == '__main__':
	main()