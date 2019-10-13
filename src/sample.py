import os, time, collections

import numpy as np
import pandas as pd
import pymc3 as pm
import theano.tensor as tt
import matplotlib.pyplot as plt

PLT = False

def main():
	np.random.seed(12345)

	K = 3
	N = 500
	spread = 5
	centers = np.array([-spread, 0, spread])

	v = np.random.randint(0, K, N)
	data = centers[v] + np.random.randn(N)

	if PLT: plt.hist(data); plt.show()

	model = pm.Model()
	with model:
		# cluster size
		pi = pm.Dirichlet('pi', a=np.ones(K), shape=K)
		# ensure all clusters have same points
		pi_min_potential = pm.Potential('pi_min_potential', tt.switch(tt.min(pi) < 0.1, -np.inf, 0))

		# cluster centers -> multivatiate
		means = pm.Normal('means', mu=np.zeros(K), sd=15, shape=K)
		# break symmetry
		order_means_potential = pm.Potential('order_means_potential', tt.switch(means[1]-means[0] < 0, -np.inf, 0) + tt.switch(means[2]-means[1] < 0, -np.inf, 0))

		# measurement error
		sd = pm.Uniform('sd', lower=0, upper=20)

		# latent cluster of each observation
		category = pm.Categorical('category', p=pi, shape=N)

		# likelihood for each observed value
		points = pm.Normal('obs', mu=means[category], sd=sd, observed=data)


	with model:
		step1 = pm.Metropolis(vars=[pi, sd, means])
		step2 = pm.ElemwiseCategorical(vars=[category], values=[0,1,2])
		tr = pm.sample(10000, step=[step1,step2])

	plt.figure()
	pm.plots.traceplot(tr, ['pi', 'sd', 'means'])
	plt.show()
	plt.close()

if __name__ == '__main__':
	main()