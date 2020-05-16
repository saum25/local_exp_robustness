'''
Created on 13 Apr 2019

@author: Saumitra
'''

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
mpl.rc('font', size=6) # code to change the fontsize of x-axis ticks when plotting x-axis in the matplotlib scientific format. (Fig. 4.8 in the thesis.)
import librosa.display as disp
import seaborn as sns
import pandas as pd
from collections import Counter

def getNumParams(tensors):
    return np.sum([np.prod(t.get_shape().as_list()) for t in tensors])

def getTrainableVariables():
    return [v for v in tf.trainable_variables()]

def read_meanstd_file(file_path):
    """
    load mean and std dev per dimension (frequency band) calculated over the Jamendo training data
    @param: file path to the mean std dev file
    @return: mean: mean across each freq band
    @return: istd: inverse of std dev across each freq band
    """   
    with np.load(file_path) as f:
        mean = f['mean']
        std = f['std']      
    istd = np.reciprocal(std)
    
    return mean, istd

def normalise(x):
    """
    Normalise an input vector/ matrix in the range 0 - 1
    @param: x: input vector/matrix
    @return: normalised vector/matrix
    """
    if x.max() == x.min(): # wierd case in the SGD optimisation, where in an intermediate step this happens
        return x
    else:
        return((x-x.min())/(x.max()-x.min()))

def save_mel(inp_mel, res_dir, prob=None, norm = True, fill_val = None, cm = 'coolwarm'):
    '''
    save input
    @param: inp_mel: input mel spectrogram
    @param: res_dir: path to save results
    @param: prob: prediction the model applies to the instance
    @param: norm: if True, normalise the input to 0-1 before saving.
    @return: NA
    '''
    plt.figure(figsize=(6, 4))
    if norm:
        disp.specshow(normalise(inp_mel), x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = cm)
    else:
        disp.specshow(inp_mel, x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = cm)
    if prob is not None: # save input
        plt.title('mel spectrogram')
        plt.tight_layout()
        plt.colorbar()
        plt.savefig(res_dir+'/'+'inp_mel'+ '_pred_'+"%.3f" %prob + '.pdf', dpi=300)
    else: # save explanation
        plt.title('explanation')
        plt.tight_layout()
        plt.colorbar()
        if type(fill_val) is str:
            plt.savefig(res_dir+'/'+'exp_fill_val_'+ fill_val +'.pdf', dpi=300)
        else:
            plt.savefig(res_dir+'/'+'exp_fill_val_'+ '%.3f' %fill_val +'.pdf', dpi=300)
        
    plt.close()
    
def plot_unique_components(unique_comps, n_samples, res_dir):
    
    '''
    Code to plot figure 4.7 in the thesis.
    It plots the distribution of the number of unigue elements in a SLIME explanation
    after multiple iterations of applying SLIME to the same instance
    The code plots results for instances from the Jamendo and RWC dataset
    with SVD-TF single neuron model as the predictor. This is referred to as exp_1 in the ppt.
    '''
       
    num_u_comps_1 = 0 # Jamendo
    num_u_comps_2 = 0 # RWC
    i = 0
    fs1 = 6
    fs2 = 17#10
    
    # calculate the number of unique components by aggregating length of all the lists
    for u_comps in unique_comps[0][0:9]: # 0 -> Jamendo results, idx = 0 to idx = 8 are the unique elements, idx=9 are the time durations
        num_u_comps_1 += len(u_comps)
    print("number of elements (Jamendo): %d" %num_u_comps_1)
    
    # calculate number of unique components by aggregating length of all the lists
    for u_comps in unique_comps[1][0:9]: # 1 -> RWC
        num_u_comps_2 += len(u_comps)
    print("number of elements (RWC): %d" %num_u_comps_2)

    
    # create a 2-d matrix, column 0 -> num_samples 2000 for e.g., column 1 -> num unique elements
    data_array_1 = np.zeros((num_u_comps_1, 2))
    print("data_array_1 shape:"),
    print data_array_1.shape
    data_array_2 = np.zeros((num_u_comps_2, 2))
    print("data_array_2 shape:"),
    print data_array_2.shape
    
    # filling the data from the Jamendo exps
    for s, u in zip(n_samples, unique_comps[0][0:9]):
        data_array_1[i:i+len(u), 0]=s 
        data_array_1[i:i+len(u), 1]=u
        i += len(u)

    i = 0
    for s, u in zip(n_samples, unique_comps[1][0:9]): # RWC data
        data_array_2[i:i+len(u), 0]=s 
        data_array_2[i:i+len(u), 1]=u
        i += len(u)    
    
    # create a pandas data frame as seaborn expects one
    # for exp 1_1 - temporal Jamendo
    df_acts_1 = pd.DataFrame(data_array_1, columns=['n_samples', 'n_unique_comps'])
    df_acts_1.n_samples = df_acts_1['n_samples'].astype('int') # change the dtype
    df_acts_1.n_unique_comps = df_acts_1['n_unique_comps'].astype('int') # change the dtype
    df_acts_1.to_csv(res_dir + '/exp1_'+str(1) + '/' + 'exp1_'+ str(1) + '_n_samp_analysis.csv', index=False)

    df_acts_2 = pd.DataFrame(data_array_2, columns=['n_samples', 'n_unique_comps'])
    df_acts_2.n_samples = df_acts_2['n_samples'].astype('int') # change the dtype
    df_acts_2.n_unique_comps = df_acts_2['n_unique_comps'].astype('int') # change the dtype
    df_acts_2.to_csv(res_dir + '/exp1_'+str(2) + '/' + 'exp1_'+ str(2) + '_n_samp_analysis.csv', index=False)

    # plotting the time taken for top-3 explanations for Jamendo - temporal    
    plt.figure(figsize = (4, 1))
    plt.plot(n_samples, unique_comps[0][9], marker='o', markersize= 3, linestyle='--', linewidth= 1)
    plt.xticks(fontsize=fs1)
    plt.ticklabel_format(style='sci', axis='x', scilimits=(0,0)) # to plot x-axis on the scientific scale
    plt.yticks(fontsize=fs1)
    plt.xlabel("Number of samples " r'($N_s$)', fontsize=fs1)
    plt.ylabel(r'$T_s$' ' (seconds)', fontsize=fs1)
    plt.savefig(res_dir + 'exp1_time.pdf', dpi=300, bbox_inches='tight')

    # plotting the distribution of unique components for different Ns values
    # exp1_1 - temporal Jamendo

    plt.figure(figsize=(10, 7))

    sns.set(color_codes=True)
    plt.subplot(211)
    sns.violinplot(x='n_samples', y='n_unique_comps', data=df_acts_1)
    plt.ylabel(r'$U_n$', fontsize=fs2)
    plt.xticks([]) # turns off x-axis ticks
    plt.xlabel('')
    plt.yticks(fontsize=fs2)
    plt.title('(a)', fontsize=fs2)
    
    # exp1_2 - temporal RWC
    plt.subplot(212)
    sns.violinplot(x='n_samples', y='n_unique_comps', data=df_acts_2)
    plt.ylabel(r'$U_n$', fontsize=fs2)
    plt.xlabel('Number of samples ' r'($N_s$)', fontsize=fs2)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.title('(b)', fontsize=fs2)
    plt.savefig(res_dir + 'exp1_n_samp_analysis.pdf', dpi=300, bbox_inches = 'tight')
    
def analyse_fv_diff(data_to_analyse):
    """
    This function computes the cardinality of the intersection set calculated 
    using explanations from a base fill value = 0 and another fill value = e.g., noise
    data_to_analyse: list of list of lists
    """
    print("explanations are for %d instances" %len(data_to_analyse))
    res = []

    for inst_exps in data_to_analyse:
        exp_base = inst_exps[0] # corresponds to fv = 0
        exp_inter_pins = []
        for exp in inst_exps:
            exp_inter_pins.append(len(set(exp) & set(exp_base)))
        res.append(exp_inter_pins)
        exp_inter_pins = []
        
    return res

def analyse_fv_diff_exp4(groundtruth, data_to_analyse):
    """
    This function computes the cardinality of the intersection set calculated 
    using explanations from a base fill value = 0 and another fill value = e.g., noise
    groundtruth: location (super-samples) of vocals
    data_to_analyse: list of list of lists
    """
    print("explanations are for %d instances" %len(data_to_analyse))
    res = []

    for inst_exps, gt in zip(data_to_analyse, groundtruth):
        exp_base = gt # corresponds to ground-truth
        exp_inter_pins = []
        for exp in inst_exps:
            exp_inter_pins.append(len(set(exp) & set(exp_base)))
        res.append(exp_inter_pins)
        exp_inter_pins = []
        
    return res


def plot_fv_senstivity(fv_exps_its, res_dir):
    """
    This function plots the cardinality of the intersection set
    w.r.t. each pair-wise computation id.
    fv_exps_its: list of list of list of lists. Each element in the outer-most list
    corresponds to one experiment, i.e., results for exp2_1
    res_dir: results directory
    Plots Fig. 4.10 in the thesis.
    """

    print("Visualisation data:"),
    print(fv_exps_its)
    
    # data for how many experiments
    print("Data from %d experiments" %(len(fv_exps_its)))
    
    # number of instances (excerpt) for the Jamendo exps: index 0, and 1 correspond to that
    n_instances_jam = len(fv_exps_its[0])
    print("Number of instances in Jamendo-based experiments: %d" %n_instances_jam)
    
    # number of instances (excerpt) for the Jamendo exps: index 2, and 3 correspond to that
    n_instances_rwc = len(fv_exps_its[2])
    print("Number of instances in RWC-based experiment: %d" %n_instances_rwc)    
    
    # create a 2-d matrix, column 0 -> comparison id, e.g., 1,  column 1 -> number of common explanations per instance for that comparison id
    data_array_jam_exp1 = np.zeros((n_instances_jam*(len(fv_exps_its[0][0])-2), 2)) # did -2 as I am not using information from min_data and zero content types. New way of plotting the figure
    print("data_array shape [Jamendo]:"),
    print data_array_jam_exp1.shape
    data_array_jam_exp2 = np.zeros((n_instances_jam*(len(fv_exps_its[0][0])-2), 2))


    data_array_rwc_exp1 = np.zeros((n_instances_rwc*(len(fv_exps_its[0][2])-2), 2))
    print("data_array shape [RWC]:"),
    print data_array_rwc_exp1.shape
    data_array_rwc_exp2 = np.zeros((n_instances_rwc*(len(fv_exps_its[0][2])-2), 2))
    
    # labels are same for both cases as the number of fvs are the same
    comp_ids = np.arange(2, len(fv_exps_its[0][0])).tolist() # we also ignore the zero by zero comparison # just 4 compid's we ignore min(data)
    print("Comparison ids:"),
    print comp_ids

    # for the ease of plotting, for each experiment, aggregate data for one comparison id together    
    exps_its=[]
    exps_final = []
    for res_exp in fv_exps_its:
        for x in [2, 3, 4]: #we ignore the zero by zero comparison for better readibility #[0, 2, 3, 4]: #range(len(res_exp[0])): # just for four cases, we ignore min(data)
            exps_its.append([d[x] for d in res_exp])
        exps_final.append(exps_its)
        exps_its = []
    print("rearranged data:"),
    print(exps_final)
    
    i = 0
    # fill exp 1 data
    for c_id, exps in zip(comp_ids, exps_final[0]):
        data_array_jam_exp1[i:i+len(exps), 0]=c_id 
        data_array_jam_exp1[i:i+len(exps), 1]=exps
        i += len(exps)

    #print data_array_jam_exp1 
    i = 0
    # fill exp 2 data
    for c_id, exps in zip(comp_ids, exps_final[1]):
        data_array_jam_exp2[i:i+len(exps), 0]=c_id 
        data_array_jam_exp2[i:i+len(exps), 1]=exps
        i += len(exps)

    #print data_array_jam_exp2
    #print('%d'%np.sum(data_array_jam_exp1 == data_array_jam_exp2))
    i = 0
    # fill exp 1 data
    for c_id, exps in zip(comp_ids, exps_final[2]):
        data_array_rwc_exp1[i:i+len(exps), 0]=c_id 
        data_array_rwc_exp1[i:i+len(exps), 1]=exps
        i += len(exps)

    #print data_array_rwc_exp1     
    i = 0
    # fill exp 2 data
    for c_id, exps in zip(comp_ids, exps_final[3]):
        data_array_rwc_exp2[i:i+len(exps), 0]=c_id 
        data_array_rwc_exp2[i:i+len(exps), 1]=exps
        i += len(exps)

    #print data_array_rwc_exp2
    
    # create a pandas data frame as seaborn expects one
    # exp2_1 -> Jamendo temporal
    df_acts_exp1 = pd.DataFrame(data_array_jam_exp1, columns=['fv_comp_ids', 'n_common_exps'])
    #df_acts_exp1['fv_comp_ids'] = df_acts_exp1['fv_comp_ids'].map({1: r'$C_1$', 2:r'$C_2$', 3: r'$C_3$', 4:r'$C_4$'})#, 5:r'$C_5$'})
    df_acts_exp1['fv_comp_ids'] = df_acts_exp1['fv_comp_ids'].map({2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4: r'$N^{norm}_g$'})
    df_acts_exp1.n_common_exps = df_acts_exp1['n_common_exps'].astype('int') # change the dtype
    df_acts_exp1.to_csv(res_dir + 'exp2_'+str(1) + '/' + 'exp2_'+ str(1) + '_fv_exps_its.csv', index=False)


    # exp2_2 -> Jamendo spectral
    df_acts_exp2 = pd.DataFrame(data_array_jam_exp2, columns=['fv_comp_ids', 'n_common_exps'])
    #df_acts_exp2['fv_comp_ids'] = df_acts_exp2['fv_comp_ids'].map({1: r'$C_1$', 2:r'$C_2$', 3: r'$C_3$', 4:r'$C_4$'})#, 5:r'$C_5$'})
    df_acts_exp2['fv_comp_ids'] = df_acts_exp2['fv_comp_ids'].map({2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4: r'$N^{norm}_g$'})
    df_acts_exp2.n_common_exps = df_acts_exp2['n_common_exps'].astype('int') # change the dtype
    df_acts_exp2.to_csv(res_dir + 'exp2_'+str(2) + '/' + 'exp2_'+ str(2) + '_fv_exps_its.csv', index=False)

    # exp2_3 -> RWC temporal
    df_acts_exp3 = pd.DataFrame(data_array_rwc_exp1, columns=['fv_comp_ids', 'n_common_exps'])
    #df_acts_exp3['fv_comp_ids'] = df_acts_exp3['fv_comp_ids'].map({1: r'$C_1$', 2:r'$C_2$', 3: r'$C_3$', 4:r'$C_4$'})#, 5:r'$C_5$'})
    df_acts_exp3['fv_comp_ids'] = df_acts_exp3['fv_comp_ids'].map({2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4: r'$N^{norm}_g$'})
    df_acts_exp3.n_common_exps = df_acts_exp3['n_common_exps'].astype('int') # change the dtype
    df_acts_exp3.to_csv(res_dir + 'exp2_'+str(3) + '/' + 'exp2_'+ str(3) + '_fv_exps_its.csv', index=False)


    # exp2_4 -> RWC spectral
    df_acts_exp4 = pd.DataFrame(data_array_rwc_exp2, columns=['fv_comp_ids', 'n_common_exps'])
    #df_acts_exp4['fv_comp_ids'] = df_acts_exp4['fv_comp_ids'].map({1: r'$C_1$', 2:r'$C_2$', 3: r'$C_3$', 4:r'$C_4$'})#, 5:r'$C_5$'})
    df_acts_exp4['fv_comp_ids'] = df_acts_exp4['fv_comp_ids'].map({2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4: r'$N^{norm}_g$'})
    df_acts_exp4.n_common_exps = df_acts_exp4['n_common_exps'].astype('int') # change the dtype
    df_acts_exp4.to_csv(res_dir + 'exp2_'+str(4) + '/' + 'exp2_'+ str(4) + '_fv_exps_its.csv', index=False)
        
    # plotting the distribution of neurons
    sns.set(color_codes=True)
    plt.figure(figsize=(7, 6))
    fs=12
    plt.subplot(2, 2, 1)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts_exp1)    
    plt.title('(a)', fontsize=fs)
    plt.xticks([]) # turns off x-axis ticks
    plt.yticks(fontsize=fs)
    plt.xlabel('')
    plt.ylabel(r'$N_{ce}$', fontsize=fs)

    plt.subplot(2, 2, 2)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts_exp2)    
    plt.title('(b)', fontsize=fs)
    plt.xticks([]) # turns off x-axis ticks
    plt.yticks(fontsize=fs)
    plt.xlabel('')
    plt.ylabel(r'$N_{ce}$', fontsize=fs)
    
    plt.subplot(2, 2, 3)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts_exp3)    
    plt.title('(c)', fontsize=fs)
    plt.ylabel(r'$N_{ce}$', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlabel('Content type', fontsize=fs)
    plt.xticks(fontsize=fs)
    
    plt.subplot(2, 2, 4)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts_exp4)    
    plt.title('(d)', fontsize=fs)
    plt.xlabel('Content type', fontsize=fs)
    plt.ylabel(r'$N_{ce}$', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)

    plt.tight_layout()
    plt.savefig(res_dir + 'slime_fv_exps_its.pdf', dpi=300, bbox_inches = 'tight')
    
    
def plot_fv_senstivity_exp4(fv_exps_its, res_dir):
    """
    This function plots the cardinality of the intersection set
    w.r.t. each pair-wise computation id.
    fv_exps_its: list of list of list of lists. Each element in the outer-most list
    corresponds to one experiment, i.e., results for exp2_1
    res_dir: results directory
    Plots Fig. 4.10 in the thesis.
    """

    print("Visualisation data:"),
    print(fv_exps_its)
    
    # number of instances (excerpt)
    n_instances = len(fv_exps_its)
    print("Number of instances in experiment: %d" %n_instances)
        
    # create a 2-d matrix, column 0 -> comparison id, e.g., 1,  column 1 -> number of common explanations per instance for that comparison id
    data_array_exp1 = np.zeros((n_instances*len(fv_exps_its[0]), 2))
    print("data_array shape:"),
    print data_array_exp1.shape
    
    # labels are same for both cases as the number of fvs are the same
    comp_ids = np.arange(1, len(fv_exps_its[0])).tolist() # just 4 compid's we ignore min(data)
    print("Comparison ids:"),
    print comp_ids

    # for the ease of plotting, for each experiment, aggregate data for one comparison id together    
    exps_final = []
    #for res_exp in fv_exps_its:
    for x in [0, 2, 3, 4]: #range(len(res_exp[0])): # just for four cases, we ignore min(data)
        exps_final.append([d[x] for d in fv_exps_its])
    print("rearranged data:"),
    print(exps_final)
    
    print("mean c1:%f" %(np.mean(exps_final[0])))
    print("mean c2:%f" %(np.mean(exps_final[1])))
    print("mean c3:%f" %(np.mean(exps_final[2])))
    print("mean c4:%f" %(np.mean(exps_final[3])))
    print(Counter(exps_final[0]))
    print(Counter(exps_final[1]))
    print(Counter(exps_final[2]))
    print(Counter(exps_final[3]))
    
    i = 0
    # fill exp 1 data
    for c_id, exps in zip(comp_ids, exps_final):
        data_array_exp1[i:i+len(exps), 0]=c_id 
        data_array_exp1[i:i+len(exps), 1]=exps
        i += len(exps)

    #print data_array_jam_exp1 
    
    # create a pandas data frame as seaborn expects one
    # exp2_1 -> Jamendo temporal
    df_acts_exp1 = pd.DataFrame(data_array_exp1, columns=['fv_comp_ids', 'n_common_exps'])
    #df_acts_exp1['fv_comp_ids'] = df_acts_exp1['fv_comp_ids'].map({1: r'$C_{G1}$', 2:r'$C_{G2}$', 3: r'$C_{G3}$', 4:r'$C_{G4}$'})#, 5:r'$C_5$'})
    df_acts_exp1['fv_comp_ids'] = df_acts_exp1['fv_comp_ids'].map({1: r'$zero$', 2:r'$min_{inp}$', 3: r'$mean_{inp}$', 4:r'$N_g$'})#, 5:r'$C_5$'})
    df_acts_exp1.n_common_exps = df_acts_exp1['n_common_exps'].astype('int') # change the dtype
    df_acts_exp1.to_csv(res_dir + 'exp4_'+str(1) + '/' + 'exp4_'+ str(1) + '_fv_exps_its.csv', index=False)
        
    # plotting the distribution of neurons
    sns.set(color_codes=True)
    plt.figure(figsize=(6, 5))
    fs=11
    plt.subplot(2, 1, 1)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts_exp1)    
    plt.ylabel(r'$N_{ce}$', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks([]) # turns off x-axis ticks
    plt.xlabel('')

    #plt.tight_layout()
    #plt.savefig(res_dir + 'slime_fv_exps_its.pdf', dpi=300, bbox = 'tight')
    
    # plotting the means
    #plt.figure(figsize=(8,6))
    plt.subplot(2, 1, 2)
    #x = [r'$C_{G1}$', r'$C_{G2}$', r'$C_{G3}$', r'$C_{G4}$']
    x = [r'$zero$', r'$min_{inp}$', r'$mean_{inp}$', r'$N^{norm}_g$']
    #means = [2.04, 1.67, 2.16, 1.96]
    top3 = [157/656.0, 47/656.0, 219/656.0, 121/656.0]
    top2 = [373/656.0, 364/656.0, 330/656.0, 404/656.0]
    top1 = [123/656.0, 233/656.0, 102/656.0, 121/656.0]
    top0 = [3/656.0, 12/656.0, 5/656.0, 10/656.0]
    
    x_pos = [i for i, _ in enumerate(x)]

    plt.bar(x_pos, top0, color='darkorchid', alpha = 1, label = r'$N_{ce}=0$', bottom = np.add(np.add(top2, top3), top1), width=0.6)
    plt.bar(x_pos, top1, color='darkcyan', alpha = 0.75, label = r'$N_{ce}=1$', bottom = np.add(top2, top3), width=0.6)
    plt.bar(x_pos, top2, color='darkkhaki', alpha = 0.75, label = r'$N_{ce}=2$', bottom = top3, width=0.6)
    plt.bar(x_pos, top3, color='darksalmon', alpha = 0.75, label = r'$N_{ce}=3$', width=0.6)

    plt.xticks(x_pos, x, fontsize=fs)
    plt.ylabel(r'$N_{instances}$', fontsize=fs)
    plt.xlabel("Content type", fontsize=fs)
    #plt.xlabel("Comparison labels")
    plt.legend(loc="best", fontsize=fs-2)
    plt.tight_layout()
    plt.savefig(res_dir + 'means.pdf', dpi=300, bbox='tight')
    

def process_exps(explanations, fvs, iterations):
    """
    code to post-process the exps from exp3
    to create a list of lists, where each 
    element of the outer list corresponds to
    $U_n$ for each instance for fv1 and so on    
    """
    
    n_fv = fvs
    iterations = iterations
    fv_indices = np.arange(0, n_fv, 1)
    n_unique_pi_pfv = []
    n_unique_comp = []
    n_unique_per_instance = []
    
    # from the input list of lists of lists make list of lists
    for exps_inst in explanations:
        for i in fv_indices:
            temp = np.arange(i, (n_fv * iterations), n_fv)
            print temp
            for j in temp:
                n_unique_pi_pfv.extend(exps_inst[j])
            n_unique_comp.append(np.unique(n_unique_pi_pfv).shape[0])
            n_unique_pi_pfv = []
        n_unique_per_instance.append(n_unique_comp)
        n_unique_comp =[]
    print n_unique_per_instance

    # rearrange the list of lists to generate each element in
    # the list belong to one fv    
    uc_per_fv = []
    uc_final =[]
    
    for i in fv_indices:
        for exps_temp in n_unique_per_instance:
            uc_per_fv.append(exps_temp[i])
        uc_final.append(uc_per_fv)
        uc_per_fv = []
    print uc_final
    return uc_final

def plot_exp3(explanations, result_path):
    """
    Plots the fig 4.9 in the thesis.
    explanations: list of lists of 5-dimensional lists, where each dim
    corresponds to a fv.
    result_path: save the fig at this directory    
    """
    
    # we use seaborn for plotting, thus data needs to be in
    # a 2-d array. column 0 -> fv id, and column 1 -> corresponding Un
    # in the input list jamendo corresponds to ids 0 and 1, and RWC to ids 2, 3
    n_instances_jam = len(explanations[0][0])
    n_instances_rwc = len(explanations[2][0])

    print("number of instances jamendo: %d rwc: %d" %(n_instances_jam, n_instances_rwc))

    n_fv = len(explanations[0]) # will be same for both the datasets
    
    # create four data arrays
    data_array_jam_exp1 = np.zeros((n_instances_jam * n_fv, 2))
    data_array_jam_exp2 = np.zeros((n_instances_jam * n_fv, 2))
    data_array_rwc_exp1 = np.zeros((n_instances_rwc * n_fv, 2))
    data_array_rwc_exp2 = np.zeros((n_instances_rwc * n_fv, 2))
    
    print("data arrays jamendo exp1: (%d, %d)" %(data_array_jam_exp1.shape))
    print("data arrays jamendo exp2: (%d, %d)" %(data_array_jam_exp2.shape))
    print("data arrays rwc exp1: (%d, %d)" %(data_array_rwc_exp1.shape))
    print("data arrays rwc exp2: (%d, %d)" %(data_array_rwc_exp2.shape))

    idx = 0
    fv_list = np.arange(0, n_fv, 1)
    
    # exp3_1 => jamendo temporal    
    for i, j in zip(fv_list, explanations[0]):
        data_array_jam_exp1[idx:idx+n_instances_jam, 0] = i
        data_array_jam_exp1[idx:idx+n_instances_jam, 1] = j
        idx += n_instances_jam

    idx = 0
    # exp3_2 => jamendo spectral    
    for i, j in zip(fv_list, explanations[1]):
        data_array_jam_exp2[idx:idx+n_instances_jam, 0] = i
        data_array_jam_exp2[idx:idx+n_instances_jam, 1] = j
        idx += n_instances_jam

    idx = 0
    # exp3_2 => rwc temporal    
    for i, j in zip(fv_list, explanations[2]):
        data_array_rwc_exp1[idx:idx+n_instances_rwc, 0] = i
        data_array_rwc_exp1[idx:idx+n_instances_rwc, 1] = j
        idx += n_instances_rwc

    idx = 0
    # exp3_2 => rwc spectral    
    for i, j in zip(fv_list, explanations[3]):
        data_array_rwc_exp2[idx:idx+n_instances_rwc, 0] = i
        data_array_rwc_exp2[idx:idx+n_instances_rwc, 1] = j
        idx += n_instances_rwc
    
    # create a pandas data frame as seaborn expects one
    # for exp 3_1 - temporal Jamendo
    df_acts_jam_exp1 = pd.DataFrame(data_array_jam_exp1, columns=['fill_value', 'n_unique_comps'])
    df_acts_jam_exp1.fill_value = df_acts_jam_exp1['fill_value'].astype('int') # change the dtype
    #df_acts_jam_exp1['fill_value'] = df_acts_jam_exp1['fill_value'].map({0: r'$Zero$', 1:r'$Min_{dataset}$', 2: r'$Min_{input}$', 3:r'$Mean_{input}$', 4:r'$Noise_{Gaussian}$'})
    df_acts_jam_exp1.n_unique_comps = df_acts_jam_exp1['n_unique_comps'].astype('int') # change the dtype
    df_acts_jam_exp1.to_csv(result_path + 'exp3_'+str(1) + '/' + 'exp3_'+ str(1) + '_fv_analysis.csv', index=False)

    # for exp 3_2 - spectral jamendo
    df_acts_jam_exp2 = pd.DataFrame(data_array_jam_exp2, columns=['fill_value', 'n_unique_comps'])
    df_acts_jam_exp2.fill_value = df_acts_jam_exp2['fill_value'].astype('int') # change the dtype
    #df_acts_jam_exp2['fill_value'] = df_acts_jam_exp2['fill_value'].map({0: r'$Zero$', 1:r'$Min_{dataset}$', 2: r'$Min_{input}$', 3:r'$Mean_{input}$', 4:r'$Noise_{Gaussian}$'})
    df_acts_jam_exp2.n_unique_comps = df_acts_jam_exp2['n_unique_comps'].astype('int') # change the dtype
    df_acts_jam_exp2.to_csv(result_path + 'exp3_'+str(2) + '/' + 'exp3_'+ str(2) + '_fv_analysis.csv', index=False)

    # for exp 3_3 - temporal rwc
    df_acts_rwc_exp1 = pd.DataFrame(data_array_rwc_exp1, columns=['fill_value', 'n_unique_comps'])
    df_acts_rwc_exp1.fill_value = df_acts_rwc_exp1['fill_value'].astype('int') # change the dtype
    df_acts_rwc_exp1['fill_value'] = df_acts_rwc_exp1['fill_value'].map({0: r'$zero$', 1:r'$min_{data}$', 2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4:r'$N^{norm}_g$'})
    df_acts_rwc_exp1.n_unique_comps = df_acts_rwc_exp1['n_unique_comps'].astype('int') # change the dtype
    df_acts_rwc_exp1.to_csv(result_path + 'exp3_'+str(3) + '/' + 'exp3_'+ str(3) + '_fv_analysis.csv', index=False)

    # for exp 3_4 - spectral rwc
    df_acts_rwc_exp2 = pd.DataFrame(data_array_rwc_exp2, columns=['fill_value', 'n_unique_comps'])
    df_acts_rwc_exp2.fill_value = df_acts_rwc_exp2['fill_value'].astype('int') # change the dtype
    df_acts_rwc_exp2['fill_value'] = df_acts_rwc_exp2['fill_value'].map({0: r'$zero$', 1:r'$min_{data}$', 2: r'$min_{inp}$', 3:r'$mean_{inp}$', 4:r'$N^{norm}_g$'})
    df_acts_rwc_exp2.n_unique_comps = df_acts_rwc_exp2['n_unique_comps'].astype('int') # change the dtype
    df_acts_rwc_exp2.to_csv(result_path + 'exp3_'+str(4) + '/' + 'exp3_'+ str(4) + '_fv_analysis.csv', index=False)
    
    sns.set(color_codes=True)
    plt.figure(figsize=(8, 6))
    fs = 12
    
    plt.subplot(2, 2, 1)    
    sns.violinplot(x='fill_value', y='n_unique_comps', data=df_acts_jam_exp1)
    plt.title('(a)', fontsize=fs)
    plt.ylabel(r'$U_n$', fontsize=fs)
    plt.xticks([]) # turns off x-axis ticks
    plt.xlabel('')
    plt.yticks(fontsize=fs)

    plt.subplot(2, 2, 2)
    sns.violinplot(x='fill_value', y='n_unique_comps', data=df_acts_jam_exp2)    
    plt.title('(b)', fontsize=fs)
    plt.xticks([]) # turns off x-axis ticks
    plt.xlabel('')
    plt.ylabel(r'$U_n$', fontsize=fs)
    plt.yticks(fontsize=fs)
    
    plt.subplot(2, 2, 3)
    sns.violinplot(x='fill_value', y='n_unique_comps', data=df_acts_rwc_exp1)     
    plt.title('(c)', fontsize=fs)
    plt.ylabel(r'$U_n$', fontsize=fs)
    plt.xlabel('Content type', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)
    
    plt.subplot(2, 2, 4)
    sns.violinplot(x='fill_value', y='n_unique_comps', data=df_acts_rwc_exp2)
    plt.title('(d)', fontsize=fs)
    plt.xlabel('Content type', fontsize=fs)
    plt.ylabel(r'$U_n$', fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xticks(fontsize=fs)

    plt.tight_layout()    
    plt.savefig(result_path + 'slime_fv_analysis.pdf', dpi=300, bbox = 'tight')

def plot_segments(seg_list, res_dir, cm = 'coolwarm'):
    """
    plots the segment figure in the paper Fig. 4.9
    """
    fs = 13
    plt.figure(figsize=(8, 3))

    plt.subplot(1, 2, 1)
    disp.specshow(seg_list[0].T, x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap=cm)
    plt.ylabel('Freq(Hz)', labelpad=0.5, fontsize=fs)
    plt.xlabel('Time(sec)', labelpad=0.5, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('(A)', fontsize=fs)

    plt.subplot(1, 2, 2)
    disp.specshow(seg_list[1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap=cm)
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    #plt.ylabel('Freq(Hz)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.title('(B)', fontsize=fs)    
    
    plt.subplots_adjust(bottom=0.125, wspace=0.25, hspace=0.25)   
    cax = plt.axes([0.93, 0.11, 0.0150, 0.77])
    cbar = plt.colorbar(cax=cax, ticks=[0, 2, 4, 6, 8])
    cbar.ax.tick_params(labelsize=fs)
    plt.savefig(res_dir, dpi=300, bbox_inches = 'tight')
    
def plot_ijcnn_fig3(data_list, res_path):
    """
    Plots fig3 in the ijcnn 2020 paper
    """
    
    fs=8
    plt.figure(figsize=(10, 2))
    
    plt.subplot(1, 6, 1)
    disp.specshow(data_list[0][0].T, x_axis='time', hop_length= 315, y_axis='mel', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.ylabel('Freq(Hz)', labelpad=0.5, fontsize=fs)
    plt.xlabel('Time(sec)', labelpad=0.5, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$input$', fontsize=fs)

    plt.subplot(1, 6, 2)
    disp.specshow(data_list[0][1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$zero$', fontsize=fs)
    
    plt.subplot(1, 6, 3)
    disp.specshow(data_list[1][1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$min_{data}$', fontsize=fs)

    plt.subplot(1, 6, 4)
    disp.specshow(data_list[2][1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$min_{inp}$', fontsize=fs)

    plt.subplot(1, 6, 5)
    disp.specshow(data_list[3][1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$mean_{inp}$', fontsize=fs)
    
    plt.subplot(1, 6, 6)
    disp.specshow(data_list[4][1].T, x_axis='time', hop_length= 315, y_axis= 'off', fmin=27.5, fmax=8000, sr=22050,cmap='coolwarm')
    plt.xlabel('Time(sec)', labelpad=1, fontsize=fs)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    cbar = plt.colorbar(orientation="horizontal", pad=0.22)
    cbar.ax.tick_params(labelsize=fs)
    cbar.ax.locator_params(nbins=4)
    plt.title(r'$N^{norm}_g$', fontsize=fs)

    plt.tight_layout()
    plt.savefig(res_path, dpi=300)   




    