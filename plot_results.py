'''
Created on 24 Jun 2019

@author: Saumitra
This file plots the figures in Chapter 4 of the thesis.
'''

import pickle
import utils

exp1 = True
exp2 = False

path_exp1 = 'results/exp1/'
idxes_exp1 = [1, 2]
path_exp2 = 'results/exp2/'
idxes_exp2 = [1, 2, 3, 4]


if exp1 == True:
    path = path_exp1
    exp_idx = idxes_exp1
else:
    path = path_exp2
    exp_idx = idxes_exp2
    
N_samples = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
exps = []
path_var = path.split('/')[1]

# load the explanations
for idx in exp_idx:
    path_final = path + path_var + '_'+str(idx) + '/' + path_var + '_'+ str(idx)
    print "loading the pickle file:",
    print path_final
    with open(path_final, 'rb') as fv:
        exps.append(pickle.load(fv))

# for the exp 1, exps must be a list of 10 elements where each element is itself a list
# the first 9 are lists of unique elements one per setting of Ns, and the last one is the list that tells how much time SLIME takes for one exp for a setting of Ns

if not exp1:
    exp_intersect = []    
    for exp in exps:
        print(exp)
        print
        exp_intersect.append(utils.analyse_fv_diff(exp))

#plot results
if exp1:
    utils.plot_unique_components(exps, N_samples, path)
else:
    utils.plot_fv_senstivity(exp_intersect, path)
