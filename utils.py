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

def save_mel(inp_mel, res_dir, prob=None, norm = True, fill_val = None):
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
        disp.specshow(normalise(inp_mel), x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = 'coolwarm')
    else:
        disp.specshow(inp_mel, x_axis = 'time', y_axis='mel', sr=22050, hop_length=315, fmin=27.5, fmax=8000, cmap = 'coolwarm')
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
    fs2 = 10
    
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

    plt.figure(figsize=(8, 5))

    sns.set(color_codes=True)
    plt.subplot(211)
    sns.violinplot(x='n_samples', y='n_unique_comps', data=df_acts_1)
    plt.ylabel(r'$U_n$', fontsize=fs2)
    plt.xticks([]) # turns off x-axis ticks
    plt.xlabel('')
    plt.yticks(fontsize=fs2)
    plt.title('(a)')
    
    # exp1_2 - temporal RWC
    plt.subplot(212)
    sns.violinplot(x='n_samples', y='n_unique_comps', data=df_acts_2)
    plt.ylabel(r'$U_n$', fontsize=fs2)
    plt.xlabel('Number of samples ' r'($N_s$)', fontsize=fs2)
    plt.xticks(fontsize=fs2)
    plt.yticks(fontsize=fs2)
    plt.title('(b)')
    plt.savefig(res_dir + 'exp1_n_samp_analysis.pdf', dpi=300, bbox = 'tight')
    
def analyse_fv_diff(data_to_analyse):
    print("data: %d" %len(data_to_analyse))
    fv_base = data_to_analyse[0] # fill value 0
    res = []
    
    for fv_new in data_to_analyse[0:]:
            res.append(len(set(fv_base) & set(fv_new))) # returns the number of comman elements
    return res

def plot_fv_senstivity(fv_exps_its, res_dir):
    
    # number of instances (excerpt)
    n_instances = len(fv_exps_its)
    
    # create a 2-d matrix, column 0 -> comparison id, e.g., 1,  column 1 -> number of common explanations per instance for that comparison id
    data_array = np.zeros((n_instances*len(fv_exps_its[0]), 2))
    print("data_array shape:"),
    print data_array.shape
    
    # labels
    comp_ids = np.arange(1, len(fv_exps_its[0])+1).tolist()
    #comp_ids = ['0', 'min(dataset)', 'min(input)', 'mean(input)', 'noise']
    i = 0
    
    exps_its=[]
    for x in range(len(fv_exps_its[0])):
        exps_its.append([d[x] for d in fv_exps_its])
    print exps_its
    
    for c_id, exps in zip(comp_ids, exps_its):
        data_array[i:i+len(exps), 0]=c_id 
        data_array[i:i+len(exps), 1]=exps
        i += len(exps)

    print data_array
    
    # create a pandas data frame as seaborn expects one
    df_acts = pd.DataFrame(data_array, columns=['fv_comp_ids', 'n_common_exps'])
    df_acts['fv_comp_ids'] = df_acts['fv_comp_ids'].map({1: '0', 2:'min(dataset)', 3: 'min(input)', 4:'mean(input)', 5:'Gaussian noise'})
    print(df_acts.head())
    #df_acts.fv_comp_ids = df_acts['fv_comp_ids'].astype('int') # change the dtype
    df_acts.n_common_exps = df_acts['n_common_exps'].astype('int') # change the dtype
    df_acts.to_csv(res_dir + 'fv_exps_its.csv', index=False)
        
    # plotting the distribution of neurons
    sns.set(color_codes=True)
    #plt.subplot(211)
    #sns.boxplot(x='fv_comp_ids', y='n_common_exps', data=df_acts)
    #a = df_acts.as_matrix()[:, 1]
    #sns.kdeplot(a[800:1600])
    #sns.distplot(a[800:1600])

    '''for i in range(5):
        print np.mean(a[800*i:(i+1)*800])
        print np.std(a[800*i:(i+1)*800])'''
    
    #plt.subplot(212)
    sns.violinplot(x='fv_comp_ids', y='n_common_exps', data=df_acts)
    #sns.kdeplot(a[1600:2400])
    #sns.distplot(a[1600:2400])
    #print np.mean(a[1600:2400])
    
    #sns.kdeplot(a[2400:3200])
    #print np.mean(a[2400:3200])
    
    #sns.kdeplot(a[3200:4000])
    #print np.mean(a[3200:4000])
    #plt.grid()
    
    plt.title('Distribution of Temporal Explanations (RWC) for fv=0 vs fv=rest')
    
    plt.savefig(res_dir + 'fv_exps_its.pdf', dpi=300)


def process_exps(explanations, fvs, iter):
    """
    code to post-process the exps from exp3
    to create a list of lists, where each 
    element of the outer list corresponds to
    $U_n$ for each instance for fv1 and so on    
    """
    
    n_fv = fvs
    iterations = iter
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
    explanations: list of 5-dimensional lists, where each dim
    corresponds to a fv.
    result_path: save the fig at this directory    
    """
    
    # we use seaborn for plotting, thus data needs to be in
    # a 2-d array. column 0 -> fv id, and column 1 -> corresponding Un
    n_instances = len(explanations[0])
    n_fv = len(explanations)
    data_array = np.zeros((n_instances * n_fv, 2))

    idx = 0
    fv_list = np.arange(0, n_fv, 1)
    for i, j in zip(fv_list, explanations):
        data_array[idx:idx+n_instances, 0] = i
        data_array[idx:idx+n_instances, 1] = j
        idx += n_instances
    #print data_array
    
    # create a pandas data frame as seaborn expects one
    # for exp 3_1 - temporal Jamendo
    df_acts = pd.DataFrame(data_array, columns=['fill_value', 'n_unique_comps'])
    df_acts.fill_value = df_acts['fill_value'].astype('int') # change the dtype
    df_acts['fill_value'] = df_acts['fill_value'].map({0: '0', 1:'min(dataset)', 2: 'min(input)', 3:'mean(input)', 4:'Gaussian noise'})
    df_acts.n_unique_comps = df_acts['n_unique_comps'].astype('int') # change the dtype
    df_acts.to_csv(result_path + 'exp3_'+str(1) + '/' + 'exp3_'+ str(1) + '_fv_analysis.csv', index=False)
    
    sns.set(color_codes=True)
    sns.violinplot(x='fill_value', y='n_unique_comps', data=df_acts)
    plt.title('Distribution of the temporal explanations for different fill values')
    plt.savefig(result_path + 'exp3'+ '_fv_analysis.pdf', dpi=300)


    
    