'''
Created on 30 Nov 2019

@author: Saumitra
'''
# This code generates a synthetic dataset for selecting the optimal content type for SLIME

import argparse
import tensorflow as tf
import numpy as np
import models.svd.Classifier as classifier
import utils
from pp_audio import prepare_audio_svd
import pickle
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import librosa.display as disp
import sys
import copy
import wrapper
import random

cf_model_path = 'models/svd/model1/Jamendo_augment_mel'
meanstd_file_path = 'models/svd'
results_path = 'synth_data/'


def create_segments(image, exp_type, n_segments):
    """
    creates the segmentation mask that SLIME uses to segment an input
    """
    segments = np.empty((image.shape[0], image.shape[1]))
    
    # select the dimension to segment
    if exp_type == 'temporal':
        use_dim = image.shape[0]
    elif exp_type == 'spectral':
        use_dim = image.shape[1]
    else:
        print("explanation type is incorrect!")
        sys.exit()
    
    hop = use_dim//n_segments
    fill_gap = use_dim%(n_segments)
    #print "hop:%d fill_gap:%d" %(hop, fill_gap)
    
    if exp_type == 'temporal':
        for i in range(n_segments):
            segments[i*hop:(i+1)*hop] = i
        if fill_gap:
            segments[(i+1)*hop:((i+1)*hop)+fill_gap]=i
    
    if exp_type == 'spectral':
        for i in range(n_segments):
            segments[:, i*hop:(i+1)*hop] = i
        if fill_gap:
            segments[:, (i+1)*hop:((i+1)*hop)+fill_gap]=i
        
    return segments


def generate_mix_pi(nv_ex, v_ex, d, song_idx):
    """
    function to create mixes by replacing d['n_s'] super-samples
    in a non-vocal input by content from the vocal stem.
    nv_ex: 3-d array of all non-vocal excerpts from a non-vocal stem
    v_ex: 3-d array of all vocal excerpts from a vocal stem
    d: parameter dictionary
    song_idx: index of song corresponding the excerpts
    """
    print("non-vocal excerpts shape:"),
    print(nv_ex.shape)
    print("vocal excerpts shape:"),
    print(v_ex.shape)
    
    # randomly select d['n_inst_pf'] instances at the same time index from both the vocal and non-vocal excerpts.
    sampling_seed = song_idx
    np.random.seed(sampling_seed)
    inst_ignore = 200 # Jan's code adds silence, this index helps to ignore those frames
    excerpt_idx = np.random.randint(low=inst_ignore, high=nv_ex.shape[0]-inst_ignore, size=d['n_inst_pf'])
    print("sampled instance indices:"),
    print(excerpt_idx)

    synth_mels = []
    synth_gt = []
    
    random.seed(sampling_seed)
    
    for e_id in excerpt_idx:
        print("generating mixes for instance: %d" %(e_id))
        print("--------")
        # sample instances corresponding to the time location
        nv = nv_ex[e_id]
        v = v_ex[e_id]

        # sample 3 numbers between 0 and d['n_samp_pi'] for both the inputs
        # np.random.randint,returns duplicates hence we do not use it
        for _ in range(d['n_mixes_pi']):

            idxes_nv = random.sample(range(d['n_seg']), d['n_s'])
            print("indices to replace"),
            print(idxes_nv)
            
            segments = create_segments(nv, exp_type = 'temporal', n_segments = d['n_seg'])
            temp = copy.deepcopy(nv)
            mask = np.zeros(segments.shape).astype(bool)
            for z in idxes_nv:
                mask[segments == z] = True
            temp[mask] = v[mask] # only takes the values of indices we want to replace and put them in nv, rest all remain the same
            
            synth_mels.append(temp)
            synth_gt.append(idxes_nv)
            
            '''plt.subplot(4,1,1)
            disp.specshow((nv.T), cmap = 'coolwarm')
            plt.subplot(4,1,2)
            disp.specshow((v.T), cmap = 'coolwarm')
            plt.subplot(4,1,3)
            disp.specshow((temp.T), cmap = 'coolwarm')
            plt.subplot(4,1,4)
            disp.specshow(segments.T, cmap = 'coolwarm')
            plt.savefig(results_path+'res_'+str(j)+'.pdf', dpi=300)'''
        print("------")
    print("number of excerpts from song %d are : %d" %(song_idx, len(synth_mels)))
    return zip(synth_mels, synth_gt)


def main():
    # parser for command line arguments    
    parser = argparse.ArgumentParser(description='program to synthesise dataset for selecting the optimal content type for SLIME')

    # the first four flags are generally not used    
    parser.add_argument('--partial', default = False, action="store_true", help='if given, allows partial reading of audio files')
    parser.add_argument('--save_inp', default = False, action="store_true", help='if given, dumps the read audio')
    parser.add_argument('--off', type=float, default=0.0, help='temporal location to start a reading an audio file (sec)')
    parser.add_argument('--dur', type=float, default=3.2, help='audio segment(sec)')
    parser.add_argument('--n_inst_pf', type=int, default=5, help='number of instances (excerpts) to select from the vocal and non-vocal stems of a song')
    parser.add_argument('--n_mixes_pi', type=int, default=4, help='number of mixes to generate by mixing the audio stems corresponding to an instance')
    parser.add_argument('--n_ss', type=int, default=3, help='number of super-samples to be mixed')
    parser.add_argument('--n_seg', type=int, default=10, help='maximum number of temporal components in an explanation')
    parser.add_argument('--dataset_path', type=str, default='../deep_inversion/', help='dataset path')
    parser.add_argument('--dataset_name', type=str, default='ccMixter', help='dataset name')
    
    args = parser.parse_args()
       
    params_dict = {
               # model params    
               'out_neurons': 1,
               'layer': 'act_out_fc9',
               'model_path': cf_model_path,
               # input representation params
               'nmels': 80,
               'excerpt_size':115,
               'partial': args.partial,
               'offset':args.off,
               'duration':args.dur,
               'save_input': args.save_inp,
               'cache_spectra': None, # from Jan's code. Directory path to store the cached spectra. Disabled by default.                             
               'mean_std_fp': meanstd_file_path,
               'dataset_path': args.dataset_path,
               'dataset_name': args.dataset_name,
                # dataset generation parameters
               'n_inst_pf': args.n_inst_pf,
               'n_mixes_pi' : args.n_mixes_pi,
               'n_seg': args.n_seg,
               'n_s': args.n_ss,
                # results params
               'results_path':results_path,
               }

    print "-------------"
    print " num out_neurons: %d" % params_dict['out_neurons']
    print " layer: %s" % params_dict['layer']
    print " model dir: %s" %params_dict['model_path']
    print "-------------"
    print " n_mels: %d" % params_dict['nmels']
    print " excerpt_size: %d" % params_dict['excerpt_size']
    print " partial: %r" % params_dict['partial']
    print " offset: %f" % params_dict['offset']
    print " duration: %f" % params_dict['duration']
    print " save_input : %r" % params_dict['save_input']
    print " cache_spectra: %r" %params_dict['cache_spectra']
    print " mean_std dir: %s" %params_dict['mean_std_fp']
    print " dataset name: %s" %params_dict['dataset_name']
    print " dataset dir: %s" %params_dict['dataset_path']
    print "-------------"
    print " n_instances_pf: %d" % params_dict['n_inst_pf']
    print " n_mixes: %d" % params_dict['n_mixes_pi']
    print " n_super-samples: %d" % params_dict['n_s']
    print " n_segments: %d" % params_dict['n_seg']      
    print "-------------"
    print " results dir: %s" % params_dict['results_path']
    print "-------------"
    print "-------------"
    
    # generate excerpts from input audio
    # returns a "list" of 3d arrays where each element has shape (no. of excerpts) x excerpt_size x nmels
    # for N songs in the dataset, there are 2N audio files, as each song has a vocal and a non-vocal stem.
    # mel_spects has elements ordered as song 1 non vocal mel spect, song 1 vocal mel spect, ....
    mel_spects, _, _ = prepare_audio_svd(params_dict)
    print("number of songs: %d audio files: %d" %(len(mel_spects)/2, len(mel_spects)))
    
    # given a pair of vocal and non-vocal excerpts corresponding to a song
    # generate n_instances_pf* n_mixes mel excerpts by first selecting n_instances_pf indices in a song, and then
    # for each index mix vocal into non-vocals by selecting n_mixes random values that tell what locations to mix. .
    mixes_final = []
    for i in np.arange(0, len(mel_spects), 2):
        nv_exceprt = mel_spects[i]
        v_excerpt = mel_spects[i+1]
        mixes_final.append(generate_mix_pi(nv_exceprt, v_excerpt, params_dict, i/2)) # returns a list where each element is a tuple of 115 x 80 mel spect and its ground-truth
    print("total number of inputs generated: %d" %(len(mixes_final)*len(mixes_final[0])))
    
    # input placeholder
    inp_ph = tf.placeholder(tf.float32, shape=(None, params_dict['nmels'], params_dict['excerpt_size'], 1))
    print("input placeholder shape: %s" %(inp_ph.shape, ))

    # generate prediction
    pred = wrapper.generate_prediction(inp_ph, params_dict)
    print("prediction vector shape: %s from layer <%s>" %(pred.shape, params_dict['layer'])) # (n_batches, n_out_neurons)
    print "-------------"
    
    threshold = 0.50
    synth_mels = []
    synth_labels =[]
    
    with tf.Session() as sess:
        # load model weights
        print("----------------------------")
        classifier_vars = utils.getTrainableVariables()
        print("Classifier Vars: " + str(utils.getNumParams(classifier_vars)))
        restorer_cf = tf.train.Saver(classifier_vars)
        restorer_cf.restore(sess, cf_model_path)
        print('classifier model restored from the file ' + cf_model_path)
        print("----------------------------")

        for file_idx in range(len(mixes_final)):
            print("----------------------------")
            print("song number: %d" %(file_idx+1))
            for mel_instance_id in range(len(mixes_final[file_idx])):
                # generate prediction
                print("")
                print("++++++++++mel instance index: %d++++++++" %mel_instance_id)                
                mel_spect = mixes_final[file_idx][mel_instance_id][0] # mel_spect shape: 115 x 80
                label = mixes_final[file_idx][mel_instance_id][1]
                input_data = mel_spect[np.newaxis, :, :, np.newaxis]
                input_data = np.transpose(input_data, (0, 2, 1, 3))
                #print("Input data shape: %s" %(input_data.shape, ))
                result = sess.run(pred, feed_dict={inp_ph:input_data})
                print("prediction probability: %f" %result[0][0])
                print("ground-truth label:"),
                print(label)

                # only saving inputs that the model classified to the vocal class, i.e. prob>=0.5
                if result[0][0] >= threshold:
                    synth_mels.append(mel_spect)
                    synth_labels.append(label)

        print("number of files in the final dataset: %d out of : %d" %(len(synth_mels), len(mixes_final) * len(mixes_final[0])))
        list_to_save = zip(synth_mels, synth_labels)

        print("data saved to the pickle object"),
        #print list_to_save
        
        with open(results_path+'data', "wb") as fp:
            pickle.dump(list_to_save, fp)
        
        



if __name__== "__main__":
    main()