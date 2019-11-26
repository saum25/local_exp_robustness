'''
Created on 24 Jun 2019

@author: Saumitra
This file plots the figures in Chapter 4 of the thesis.
'''

import pickle
import utils

exp1 = False
exp2 = True
exp3 = False

path_exp1 = 'results/exp1/'
idxes_exp1 = [1, 2]
path_exp2 = 'results/exp2/'
idxes_exp2 = [1, 2, 3, 4]
path_exp3 = 'results/exp3/'
idxes_exp3 = [1, 2, 3, 4]

if exp1 == True:
    path = path_exp1
    exp_idx = idxes_exp1
elif exp2 == True:
    path = path_exp2
    exp_idx = idxes_exp2
else:
    path = path_exp3
    exp_idx = idxes_exp3
    
N_samples = [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
fvs = 5
iterations = 5
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


# for the exp 2, each element in exps is a list of list of lists
# i.e, for each instance, we generate explanations for each of the fv.
if exp2 == True:
    exp_intersect = []    
    for exp in exps:
        print('Explanations:'),
        print(exp)
        exp_intersect.append(utils.analyse_fv_diff(exp))

# for experiment 3, we have exps as the list of list of lists
# each element of the main list corresponds to one instance
# each element of a list corresponding to an instance represents Top-3 exps
# a total of 5 (num of fv's) * 5 (num_iters) exps for each instance
# ordered as exp[0][0] = instance_1_exp_fv1, exp[0][1] = instance_1_exp_fv2 and so on
exp_pp = []
if exp3 == True:
    for exp_idx in range(len(exps)):
        exp_pp.append(utils.process_exps(exps[exp_idx], fvs, iterations))

#plot results
if exp1:
    utils.plot_unique_components(exps, N_samples, path)
elif exp2:
    utils.plot_fv_senstivity(exp_intersect, path)
else:
    utils.plot_exp3(exp_pp, path)
    
    
    
    
    
    
    