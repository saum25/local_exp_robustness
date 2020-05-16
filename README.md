# IJCNN-2020

1. This repository contains the code and other required information to reproduce the results of the paper - "Reliable Local Explanations for Machine Listening", In Proceedings of IJCNN Special Session on Explainable Computational/Artificial Intelligence, Glasgow, July, 2020.

2. The details of the directories and the python files are mentioned below.
   - Directories
     - **lime**: contains the code of the LIME algorithm. At some places the original code has been modified for using it in the    experiments in this paper. 
     - **models**: contains the pre-trained Tensorflow model called as "SVDNet-R1" in the paper and the mean and standard deviation of input data for each dataset.
     - **synth_data**: contains the synthetic dataset used in Section 5 of the paper. 
   - Python files
     - audio.py
     - augment.py
     - create_test_script.py
     - datasetgen.py
     - plot_results.py
     - pp_audio.py
     - progress.py
     - simplecache.py
     - utils.py
     - wrapper.py
