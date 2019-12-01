'''
Created on 13 Apr 2019

@author: Saumitra
'''

import os
import io
import augment
from progress import progress
from simplecache import cached
import audio
import numpy as np
import utils
import pickle

synth_data_path = 'synth_data/data'

def prepare_audio_svd(parameters):
    """
    Reads input audio and creates Mel-spectrogram excerpts
    of shape 115 x 80 needed by the neural network model
    """
    if parameters['sd'] == True:
        # use the saved data
        with open(synth_data_path, 'rb') as fp:
            temp_data = pickle.load(fp)
        
        spectrum_list = [ele[0] for ele in temp_data]
        print("number of excerpts:%d" %(len(spectrum_list)))
        spectrum = [ele[np.newaxis, :, :] for ele in spectrum_list]
            
        # -read mean and 1.0/std dev per mel band
        print("mean_std_file path: %s" %(parameters['mean_std_fp'] + '/' + parameters['dataset_name'] + '_meanstd.npz'))
        mean, istd = utils.read_meanstd_file(parameters['mean_std_fp'] + '/' + parameters['dataset_name'] + '_meanstd.npz') 

    else:
        # default parameters from ISMIR 2015: Jan et. al.   
        sample_rate = 22050
        frame_len = 1024
        fps = 70
        mel_min = 27.5
        mel_max = 8000
        mel_bands = parameters['nmels']
        blocklen = parameters['excerpt_size']
        
        bin_nyquist = frame_len // 2 + 1
        bin_mel_max = bin_nyquist * 2 * mel_max // sample_rate
        
        # prepare dataset
        print("Preparing data reading...")
        datadir = os.path.join(parameters['dataset_path'], 'datasets', parameters['dataset_name'])
    
        # - load filelist
        '''with io.open(os.path.join(datadir, 'filelists', 'valid')) as f:
            filelist = [l.rstrip() for l in f if l.rstrip()]'''
        with io.open(os.path.join(datadir, 'filelists', 'test')) as f:
            filelist = [l.rstrip() for l in f if l.rstrip()]
            
        if not parameters['partial']:
            #duration and offset arguments have no use in this part of the code.
            # - create generator for spectra
            spects = (cached(parameters['cache_spectra'] and
                             os.path.join(parameters['cache_spectra'], fn + '.npy'),
                             audio.extract_spect,
                             os.path.join(datadir, 'audio', fn),
                             sample_rate, frame_len, fps)
                      for fn in filelist)
        else:        
            # - create generator for spectra
            spects = (cached(parameters['cache_spectra'] and
                             os.path.join(parameters['cache_spectra'], fn + '.npy'),
                             audio.extract_spect_partial,
                             os.path.join(datadir, 'audio', fn),
                             parameters['save_input'], parameters['results_path'], sample_rate, frame_len, fps, parameters['offset'], parameters['duration'])
                      for fn in filelist)
            
        # -read mean and 1.0/std dev per mel band
        print("mean_std_file path: %s" %(parameters['mean_std_fp'] + '/' + parameters['dataset_name'] + '_meanstd.npz'))
        mean, istd = utils.read_meanstd_file(parameters['mean_std_fp'] + '/' + parameters['dataset_name'] + '_meanstd.npz') 
    
        # - prepare mel filterbank
        filterbank = audio.create_mel_filterbank(sample_rate, frame_len, mel_bands,
                                                 mel_min, mel_max)  
        
        filterbank = filterbank[:bin_mel_max]#.astype(floatX)
        
        spects = (np.log(np.maximum(np.dot(spect[:, :bin_mel_max], filterbank), 1e-7)) for spect in spects)
            
        # - define generator for Z-scoring
        spects = ((spect - mean) * istd for spect in spects)
    
        # - define generator for silence-padding
        pad = np.tile((np.log(1e-7) - mean) * istd, (blocklen // 2, 1))
        spects = (np.concatenate((pad, spect, pad), axis=0) for spect in spects)
        
        # - we start the generator in a background thread (not required)
        spects = augment.generate_in_background([spects], num_cached=1)
        
        spectrum = []   # list of 3d arrays.each 3d array for one audio file No. of excerpts x 115 x 80
        
        print("Generating excerpts:")
        for spect in progress(spects, total=len(filelist), desc='File '):
            print("spect shape: %s" %(spect.shape, )),
            num_excerpts = len(spect) - blocklen + 1
            excerpts = np.lib.stride_tricks.as_strided(
                    spect, shape=(num_excerpts, blocklen, spect.shape[1]),
                    strides=(spect.strides[0], spect.strides[0], spect.strides[1]))
            print("mel excerpt shape: %s" %(excerpts.shape, ))
            spectrum.append(excerpts)
            
    return spectrum, mean, istd

