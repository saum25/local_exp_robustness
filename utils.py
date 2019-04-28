'''
Created on 13 Apr 2019

@author: Saumitra
'''

import numpy as np
import tensorflow as tf
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
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
        plt.savefig(res_dir+'/'+'exp_fill_val_'+ '%.3f' % fill_val +'.pdf', dpi=300)
        
    plt.close()
    
def plot_unique_components(unique_comps, n_samples, res_dir):
    
    '''plt.figure(figsize=(6,4))
    for ele, samp in zip(unique_comp_per_instance, samples):
        plt.plot(ele, marker = 'o', label = str(samp))
    plt.xlabel('instance id')
    plt.ylabel('n_unique components')
    plt.title('analysing unique components')
    plt.grid()
    plt.legend()
    plt.savefig(res_dir+'n_unique_comp.pdf', dpi=300)
    plt.close()'''
    
    num_u_comps = 0
    i = 0
    
    # calculate number of unique components by aggregating length of all the lists
    for u_comps in unique_comps:
        num_u_comps += len(u_comps)
    print("number of elements: %d" %num_u_comps)
    
    # create a 2-d matrix, column 0 -> num_samples 2000 for e.g., column 1 -> num unique elements
    data_array = np.zeros((num_u_comps, 2))
    print("data_array shape:"),
    print data_array.shape
    
    for s, u in zip(n_samples, unique_comps):
        data_array[i:i+len(u), 0]=s 
        data_array[i:i+len(u), 1]=u
        i += len(u)
    
    # create a pandas data frame as seaborn expects one
    df_acts = pd.DataFrame(data_array, columns=['n_samples', 'n_unique_comps'])
    df_acts.n_samples = df_acts['n_samples'].astype('int') # change the dtype
    df_acts.to_csv(res_dir + 'n_samp_analysis.csv', index=False)
        
    # plotting the distribution of neurons
    sns.set(color_codes=True)
    plt.subplot(211)
    sns.boxplot(x='n_samples', y='n_unique_comps', data=df_acts)
    plt.subplot(212)
    sns.violinplot(x='n_samples', y='n_unique_comps', data=df_acts)
    
    plt.savefig(res_dir + 'n_samp_analysis.pdf', dpi=300)
    
    
    
    
    
    