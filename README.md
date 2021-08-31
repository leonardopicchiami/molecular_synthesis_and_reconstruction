# MOLECULAR SYNTHESIS & RECONSTRUCTION

This is the Python repository about the Deep Learning And Applied Artificial Intelligence Project 2019/2020 in the Computer Science Master's Degree course of La Sapienza University of Rome.


## Goal Description

The goal was to develop Deep Learning models that can reconstruct molecules from low dimensional space representation and generate new molecules. Starting from the paper [[1]](#1), we chose a fixed representation of the molecules among the proposed ones; we selected a type of model (e.g., RNN, VAE, AAE, GAN ..) and, we designed several versions of the chosen model. Finally, we evaluated models by using several metrics. We aimed to improve the current state-of-the-art performances and results.

We performed the entire work using Google Colab in the notebook below.

**Title** | **Notebook** | **Notebook on Colab**
|------------ | ------------- | ------------ |
Molecular Syntesis & Reconstruction | [notebook](https://github.com/leonardopicchiami/molecular_synthesis_and_reconstruction/blob/master/Molecular%20Synthesis%20%26%20Reconstruction.ipynb) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/leonardopicchiami/molecular_synthesis_and_reconstruction/blob/master/Molecular_Synthesis_%26_Reconstruction.ipynb) |

Below is a summary of what is reported in the notebook.

The whole project is divided as follows:

- Molecular Representation & Dataset
- Preprocessing
- Models Building and Training 
- Molecules Reconstruction
- Molecules Synthesis 


We used [PyTorch](https://pytorch.org) to buid and traint each Deep Learning developed model.


### Molecular Representation & Dataset

Among the representations presented by the paper, we selected the SMILE string representation. [[1]](#1) shows several state-of-the-art works and, for each one is reported:

- the model architecture.
- the selected representation of the molecules.
- the chosen dataset.
- the number of molecules in the dataset.

In literature, many authors used the ZINC dataset to provide an experimental evaluation for the designed approach. Since the ZINC dataset contains millions of molecules, many works show a smaller and cleaned ZINC dataset (*i.e.*, a subset of the main ZINC dataset) to present new reconstruction and synthesis approaches. This dataset, called ZINC250K, contains approximately 250,000 examples. We decided to use the ZINC250K dataset for this work.



### Preprocessing

In the preprocessing phase, we designed how to transform the chosen SMILE string representation in a tensor. To avoid excessively large sparse tensors with too much padding, we selected only SMILE strings at most 60 characters since the distribution of the lengths of the strings showed that the majority of the SMILE strings present a length at most 60.

To perform this task, we split the ZINC250K in *train set* e *test set*: *train set* 70% - *test set* 30% by select only the strings with length at most 60. Clearly, the training phase was carried out only in the train set.


### Models Building and Training

We chose to use VAE (Variational Autoencoder). We implemented two different VAEs:

- Convolutional Encoder - Convolutional Decoder
- Convolutional Encoder - GRU (Multi-layer) + Linear layer

For both models, we carried out training sessions for 35 epochs with several ad-hoc hyperparameters tunings. 

The results show that, for both developed models, we can train every model for fewer epochs. This can lead to having equally efficient models.   Consequently, we chose the most efficient hyperparameter tuning for every model and, we re-trained each model for a smaller number of epochs. The results show truly efficient models.


### Molecules Reconstruction

In the reconstruction phase, we evaluated the best model (in terms of accuracy) for both types to establish its goodness in reconstructing the molecules. Reconstructing a given molecule means, firstly, encoding it in the latent space representation. Then, decoding it from the latent space representation to get the starting representation. 

The goal is to allow the model to learn the latent space as good as possible. This enables the potential for the model to perform an encoding-decoding task with an exact representation of the starting molecule as result. We evaluated every model the accuracy of the molecules reconstruction and the validity ratio of the reconstructed molecules both on the test set and train set. 

For the Convolutional Encoder - Convolutional Decoder:


| Dataset| Recontruction | Validity  
|---|---|---|
| train |  99.91%  | 99.94%  |
| test  |  96.80%  | 97.91%  |


For the Convolutional Encoder - GRU + Linear Decoder: 


| Dataset| Recontruction | Validity  
|---|---|---|
| train |  99.72%  | 99.78%  |
| test  |  93.63%  | 99.79%  |




### Molecules Synthesis

In the synthesis phase, we synthesised new molecules by sampling from the prior distribution chosen for the latent space. We used the following formula:

<p align = "center">
  <img src= "https://latex.codecogs.com/gif.latex?stdev&space;\times&space;\mathcal{N}(0,&space;1)&space;&plus;&space;latent" />

We generated new topologically similar molecules starting from a given molecule. This molecule has to be mapped in the latent space representation since we used it as a mean for the above-mentioned formula (*i.e.*, the latent operand). Basically, we perturbed it by fixing a standard deviation, called *stdev*, and sampling from the normal distribution.

We performed several experiments in this direction. In particular, we choose two different molecules and three different standard deviation values. We sampled 5000 potential molecules for each chosen starting molecule and chosen standard deviation value. We chose *Aspirine* and *Tamiflu* as starting molecules. We selected the following standard deviation values: [0.045, 0.065, 0.085].

To validate the synthesis of new molecules we planned to select molecules that are not in the ZINC250K dataset.


Given ![](https://latex.codecogs.com/gif.latex?\mathcal{G}) the set of chemically valid molecules, ![](https://latex.codecogs.com/gif.latex?\mathcal{D}) the train set, ![](https://latex.codecogs.com/gif.latex?n) the number of syntatically valid generated molecules, ![](https://latex.codecogs.com/gif.latex?n_{samp}) the number of sampling done, the following metrics were used to evaluate the generation of molecules:


  
![](https://latex.codecogs.com/gif.latex?Syntatic\&space;Validity\&space;Ratio:\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\frac{n}{n_{samp}})
  
![](https://latex.codecogs.com/gif.latex?Chemical\&space;Validity\&space;Ratio:&space;\&space;\&space;\&space;\&space;\&space;\&space;\&space;\frac{|\mathcal{G}|}{n})
   
![](https://latex.codecogs.com/gif.latex?Uniqness:&space;\vspace{0.5cm}&space;\frac{|set(\mathcal{G})|}{n})   

![](https://latex.codecogs.com/gif.latex?Novelty:\&space;\&space;\&space;\&space;1&space;-&space;\frac{|\mathcal{G}&space;\cap&space;\mathcal{D}|}{|\mathcal{G}|})

![](https://latex.codecogs.com/gif.latex?SimilarityRatio:\&space;\&space;\&space;\&space;\frac{\sum_{i=0}^{|\mathcal{G}|}&space;\sum_{j=i&plus;1}^{|\mathcal{G}|}&space;TanimotoSim(\mathcal{G}_i,&space;\mathcal{G}_j)}{\frac{|\mathcal{G}|(|\mathcal{G}|-&space;1)}{2}})

where TanimotoSim is the Tanimoto Similarity defined as:


<p align = "center">
  <img src= "https://latex.codecogs.com/gif.latex?TanimotoSim(A,&space;B)&space;=&space;\frac{A&space;\cdot&space;B}{||A||^2&space;&plus;&space;||B||^2&space;-&space;A&space;\cdot&space;B}" />


The scores are reported only in the notebook.



## Description and Requirements

Python 3 was used for the development of the system. We developed and tested the system using 64-bit Python 3.6 and 64-bit Python 3.7 on Linux Systems.  Partially, we locally developed it on Linux distros: Linux Mint and Ubuntu. However, we performed the majority of the whole work on Google Colab. Since the core of the project is in the jupyter-notebook, we do not provide specific scripts for several sub-tasks such as evaluation, train and synthesis.

As future developments, we plan to develop specific scripts that allow these computations locally or on High-Performance Computing (HPC) machines.

We used the following libraries:

- NumPy
- PyTorch (torch and torchvision)
- Scikit-Learn
- Pandas
- Tqdm
- Matplotlib
- Plotly
- RDKit

RDKit is a library for Cheminformatics and Machine Learning widely used in literature.


## References

<a id="1">[1]</a>
Elton, Daniel & Boukouvalas, Zois & Fuge, Mark & Chung, Peter. (2019). Deep learning for molecular design - a review of the state of the art. Molecular Systems Design & Engineering. 10.1039/C9ME00039A. 


## License

The license for this software is: GPLv3.
