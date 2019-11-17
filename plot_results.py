'''
Created on 24 Jun 2019

@author: Saumitra
'''

import pickle
import utils

path = 'results/exp2_1/exp2_1'
N_samples = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]

# load the explanations
with open(path, 'rb') as fv:
    exps = pickle.load(fv)

exp_intersect = []

for exp in exps:
    print(exp)
    print
    exp_intersect.append(utils.analyse_fv_diff(exp))

#plot results
utils.plot_fv_senstivity(exp_intersect, path)
#utils.plot_unique_components(exps[0:9], N_samples, path)
