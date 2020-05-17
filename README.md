# IJCNN-2020

1. This repository contains the code and other required information to reproduce the results of the paper - "Reliable Local Explanations for Machine Listening", In Proceedings of IJCNN Special Session on Explainable Computational/Artificial Intelligence, Glasgow, July, 2020.

2. The details of the directories and the python files are mentioned below.
   - Directories
     - **lime**: contains the code of the LIME algorithm. At some places the original code has been modified for using it in the    experiments in this paper. 
     - **models**: contains the pre-trained Tensorflow model called as "SVDNet-R1" in the paper and the mean and standard deviation of input data for each dataset.
     - **synth_data**: contains the synthetic dataset and ground-truth information used in Section 5 of the paper. 
   - Python files
     - **wrapper.py**: main file that contains code to read input audio, generate its SVDNet-R1 prediction, and finally explain the prediction using SLIME.
     - **pp_audio.py**, **audio.py**, **augment.py**, **simplecache.py**, and **progress.py**: files related to transforming musical audio data into mel-spectrograms inputs for generating predictions from the SVDNet-R1 model.
     - **create_test_script.py**, **datasetgen.py**: files related to the generation of synthetic dataset using the vocal and non-vocal stems from the CCMixter dataset.
     - **plot_results.py**, **utils.py**: files that contain code of some helper functions and the code to plot results of all the experiments.

3. To execute the code use the script below.

   - * * python wrapper.py --dataset_path "path_to_the_dataset" * *
  
   - The results are by default saved as a pickle file at the path "results/". Thus, before executing the code, one needs to create this directory.
   - The code has several input arguments that can be configured depending on the experiment one needs to execute. For example, to generate the Jamendo dataset results for the experiment in Section 4 (a), we need to use the script below
     - python wrapper.py --dataset_path "path_to_the_Jamendo_dataset" --n_inst_pf 25 --n_samp_exp

4. The code has been developed using frameworks mentioned in the packages.txt file.

In case of any issue in using the code, please contact smishra@turing.ac.uk
