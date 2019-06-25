'''
Created on 8 Apr 2019

@author: Saumitra
'''
# A wrapper that calls different local explanation methods
# There are three key components
# 1) a list of datasets to get instances from 
# 2) a list of pre-trained models to explain
# 3) a list of local explanation methods

import argparse
import tensorflow as tf
import numpy as np
import models.svd.Classifier as classifier
import utils
from pp_audio import prepare_audio_svd
from lime import lime_image
import time
import pickle


cf_model_path = 'models/svd/model1/Jamendo_augment_mel'
meanstd_file_path = 'models/svd/jamendo_meanstd.npz'
results_path = 'results/'

# Seems redundant - TODO -> Improve this
def prediction_fn(sess, inp_sym, pred_sym, inp_real):
    return sess.run(pred_sym, feed_dict={inp_sym:inp_real})
    
def generate_prediction(inp, parameters):
    """
    generates prediction(s) by passing the input to the classifier.
    @param: inp: input
    @param: parameters: configuration dictionary
    @return: prediction: generated prediction
    """
    
    # training mode flag
    training_mode = tf.constant(False, dtype=tf.int32)
    
    # create classifier architecture    
    classifier_network = classifier.architecture(inp, training_mode, parameters['out_neurons'])
    
    # generate prediction
    prediction = classifier_network[parameters['layer']]
    return prediction

def main():
    # parser for command line arguments    
    parser = argparse.ArgumentParser(description='program to analyse the robustness of local explanation methods for audio data')
    
    parser.add_argument('--partial', default = False, action="store_true", help='if given, allows partial reading of audio files')
    parser.add_argument('--save_inp', default = False, action="store_true", help='if given, dumps the read audio')
    parser.add_argument('--off', type=float, default=0.0, help='temporal location to start a reading an audio file (sec)')
    parser.add_argument('--dur', type=float, default=3.2, help='audio segment(sec)')
    parser.add_argument('--n_inst_pf', type=int, default=5, help='number of instances(excerpts) to explain per audio file')
    parser.add_argument('--e_type', type=str, default='temporal', help='explanation type')
    parser.add_argument('--n_seg', type=int, default=10, help='maximum number of components in an explanation')
    parser.add_argument('--iterate', type=int, default=5, help='number of SLIME iterations per instance')
    parser.add_argument('--dist_metric', type=str, default='l2', help='distance metric')
    parser.add_argument('--dataset_path', type=str, default='../deep_inversion/', help='dataset path')
    parser.add_argument('--dataset_name', type=str, default='jamendo', help='dataset name')
    parser.add_argument('--n_samp_exp', default = False, action="store_true", help='if given, runs the code in exp1 mode, i.e., finding Ns')
    
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
                # SLIME params
               'n_inst_pf': args.n_inst_pf,
               'iterate' : args.iterate,
               'dist_metric': args.dist_metric,
               'n_samp_mode': args.n_samp_exp,
               'e_type': args.e_type,
               'n_seg': args.n_seg,
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
    print " iterations: %d" % params_dict['iterate']
    print " distance_metric: %s" % params_dict['dist_metric']
    print " n_samp_mode: %r" % params_dict['n_samp_mode']
    print " explanation type: %s" % params_dict['e_type']
    print " n_segments: %d" % params_dict['n_seg']      
    print "-------------"
    print " results dir: %s" % params_dict['results_path']
    print "-------------"
    print "-------------"

    # input placeholder
    inp_ph = tf.placeholder(tf.float32, shape=(None, params_dict['nmels'], params_dict['excerpt_size'], 1))
    print("input placeholder shape: %s" %(inp_ph.shape, ))

    # generate prediction
    pred = generate_prediction(inp_ph, params_dict)
    print("prediction vector shape: %s from layer <%s>" %(pred.shape, params_dict['layer'])) # (n_batches, n_out_neurons)
    print "-------------"
    
    # generate excerpts from input audio
    # returns a "list" of 3d arrays where each element has shape (no. of excerpts) x excerpt_size x nmels
    mel_spects, mew, i_std = prepare_audio_svd(params_dict)

    # finding the minimum bin value over the dataset
    min_val = min([np.min(arr) for arr in mel_spects])
    print("minimum bin value over the dataset: %f " %min_val)
    
    # normalised noise distribution
    noise_arr = np.random.normal(0, 1, (params_dict['excerpt_size'], params_dict['nmels']))
    noise_arr_norm = (noise_arr-mew)*i_std
    
    
    if params_dict['n_samp_mode']:
        N_samples = [30, 50]#[1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000]
    else:
        N_samples = [30]#[70000]
               
    inst_ignore = 200 # approximately ignore the ones with padding
    
    agg_comps_per_instance = []
    if params_dict['n_samp_mode']:
        unique_comps_per_instance = []
        unique_comps = []
        time_list = []
        time_per_ns = []
    else:
        ins_intersection=[]
        
    with tf.Session() as sess:
        # load model weights
        print("----------------------------")
        classifier_vars = utils.getTrainableVariables()
        print("Classifier Vars: " + str(utils.getNumParams(classifier_vars)))
        restorer_cf = tf.train.Saver(classifier_vars)
        restorer_cf.restore(sess, cf_model_path)
        print('classifier model restored from the file ' + cf_model_path)
        print("----------------------------")
        
        for n_samples in N_samples:
            print("N_samples in the perturbed distribution: %d" %n_samples)
            
            for file_idx in range(len(mel_spects)):
                print("----------------------------")
                print("file number: %d" %(file_idx+1))

                sampling_seed = file_idx # result in different instance indices per file
                np.random.seed(sampling_seed)
                excerpt_idx = np.random.randint(low=inst_ignore, high=mel_spects[file_idx].shape[0]-inst_ignore, size=params_dict['n_inst_pf'])
                print("sampled instance indices:"),
                print(excerpt_idx)
                      
                for mel_instance_id in excerpt_idx:
                    # generate prediction
                    print("")
                    print("++++++++++mel instance index: %d++++++++" %mel_instance_id)                
                    mel_spect = mel_spects[file_idx][mel_instance_id] # mel_spect shape: 115 x 80
                    input_data = mel_spect[np.newaxis, :, :, np.newaxis]
                    input_data = np.transpose(input_data, (0, 2, 1, 3))
                    #print("Input data shape: %s" %(input_data.shape, ))
                    result = sess.run(pred, feed_dict={inp_ph:input_data})
                    print("prediction probability: %f" %result[0][0])
                    
                    # save the instance
                    #utils.save_mel(mel_spect.T, res_dir = results_path, prob = result[0][0], norm = False)
                    
                    # use SLIME to explain the prediction
                    if params_dict['n_samp_mode']:
                        fill_value = [0]
                    else:
                        fill_value = [0, min_val, np.min(mel_spect), np.mean(mel_spect), 'noise']
                    
                    for idx in range(args.iterate):
                        print("---iteration:%d----" %(idx+1))
                        for val in fill_value:
                            if params_dict['n_samp_mode']:
                                start = time.time()
                            print("\n fill value: %s" %val)        
                            explainer = lime_image.LimeImageExplainer(verbose=True)
                            explanation, segments = explainer.explain_instance(image = mel_spect, classifier_fn = prediction_fn, hide_color = val, 
                                                                               top_labels = 1, num_samples = n_samples, distance_metric = args.dist_metric, sess = sess, 
                                                                               inp_data_sym = inp_ph, score_sym = pred, exp_type= params_dict['e_type'], n_segments= params_dict['n_seg'], batch_size=16, noise_data = noise_arr_norm)
                            #utils.save_mel(segments.T, results_path, prob=None, norm=False, fill_val=val)
                            agg_exp, _, exp_comp_weights, pred_err = explanation.get_image_and_mask(label = 0, positive_only=False, hide_rest=True, num_features=3)
                            
                            if params_dict['n_samp_mode']:
                                exp_t = time.time()-start
                                time_list.append(exp_t)
                                print("time taken: %f" %(exp_t))
                            
                            print("SLIME explanations: "),
                            print(exp_comp_weights)
                            #print("prediction error: %f" %(pred_err))
                            #utils.save_mel(agg_exp.T, results_path, prob=None, norm= False, fill_val= val)
                            if params_dict['n_samp_mode']:
                                agg_comps_per_instance.extend([ele[0] for ele in exp_comp_weights])
                            else:
                                agg_comps_per_instance.append([ele[0] for ele in exp_comp_weights])
                    
                    print("=================================")
                    print("aggregated components per instance over %d iterations:" %args.iterate),
                    print agg_comps_per_instance
                    if params_dict['n_samp_mode']:
                        n_unique_comp = np.unique(agg_comps_per_instance).shape[0]
                        print("number of unique components: %d" %(n_unique_comp))
                        unique_comps_per_instance.append(n_unique_comp)
                    else:
                        #ins_intersection.append(utils.analyse_fv_diff(agg_comps_per_instance)) # save cardinalty of the intersection set
                        ins_intersection.append(agg_comps_per_instance) # save all explanations per instance., so ins_intersection is a list of list of lists
                    agg_comps_per_instance = []
                    
            if params_dict['n_samp_mode']:
                print("unique components over all instances for n_samples: %d:" %n_samples), 
                print unique_comps_per_instance
                unique_comps.append(unique_comps_per_instance)
                unique_comps_per_instance = []
                time_per_ns.append(np.mean(np.asarray(time_list)))
                time_list = []
                
        #save the nested list as a pickle object
        if params_dict['n_samp_mode']:
            list_to_save = unique_comps
            list_to_save.append(time_per_ns) # last list is of time
        else:
            list_to_save = ins_intersection
            
        print("data saved to the pickle object:"),
        print list_to_save
        
        with open("results/exps", "wb") as fp:
            pickle.dump(list_to_save, fp)
        
if __name__== "__main__":
    main()