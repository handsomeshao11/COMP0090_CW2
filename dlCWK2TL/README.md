## Introduction to Deep Learning (COMP0090)
Course Work 2

### Requirements
Results were generated using python 3.9.7 
with packages specified in `requirements.txt`

It is recommended to use a virtual environment with required packages installed.

To install required packages, to current (virtual) environment, use
`pip install -r requirements.txt`

### Run scripts to produce results

The scripts used to generate results for this assignments are
`data.py`,`u_net.py`, `v_unet_unablated.py`, `v_unet_ablated.py`, `r_unet.py`,
`m_unet.py`, `d_unet.py`, `prediction.py`, any of which can be run via

`python data.py`

### First Step

The first step is to download the data. This must be done before any of the models
can be built. This can be done by running

`python data.py`

The results are written to `./data_restructured`, which will be broken down into
three subfolders. These are `./data_restructured/train`, `./data_restructured/test`
and `./data_restructured/val` 


### Second Step

Once the data is downloaded, you can then build the models by running the scripts

`python u_net.py`
`python v_net_unablated.py`
`python v_net_ablated.py`
`python r_net.py`
`python m_net.py`
`python d_net.py`

The data is loaded in batches using the data loading generator in `loader.py`.
This is done from h5 file and not memory. With the function of this file, you
can specify the batch size, whether to shuffle the dataset or not and the types
of target values (e.g., masks). The instance of the class returns a tuple of train 
data and target value with the batch size. You can see a summary of the dataset 
by running `loader.py` in the command line.

To build and evaluate the models, all functions can be found in `Models.py` .

The evaluation results will then be saved in `dlCWK2TL/evaluation_results`.
The models will be saved in `dlCWK2TL/models`.


### Third Step

Once the models have been built, you can then visualise the predictions by
running the following script

`python prediction.py`

The images of these predictions will then be saved in the folder `dlCWK2TL/visual_predictions`.





