# MOLECULAR SYNTHESIS & RECONSTRUCTION

This is the Python repository about the Deep Learning And Applied Artificial Intelligence Project 2019/2020 in the Computer Science Master's Degree course of La Sapienza University of Rome.


## Goal Description

The goal was to develop Deep Learning models that can reconstruct molecules as best as possible and generate new molecules. The idea was to analyze the paper [[1]](#1): choose a representation of the molecules among the proposed one, choose a type of model and try different models of the same type (RNN, VAE, AAE, GAN ..) and choose different metrics on which evaluate models. The actual goal has been to get models that reconstruct and generate new ones in a good enough way compared to other models that are state of the art.

The entire work is reported and performed using Google Colab in the notebook below.

**Title** | **Notebook** | **Notebook on Colab**
|------------ | ------------- | ------------ |
Molecular Syntesis & Reconstruction | [notebook](.) | [![Open in Colab](.)] |

Here is a brief summary of what is reported in the notebook.

The whole project is divided into parts:
- Molecular Representation & Dataset
- Preprocessing
- Models Building and Training 
- Molecules Reconstruction
- Molecules Synthesis 



### Molecular Representation & Dataset

Among the representations presented in the paper, it was chosen to use the representation as a SMILE string. Also [[1]](#1), shows the various models present in the state of the art and for each model:

- the chosen architecture
- the representation of the molecules used
- the dataset used
- the number of examples in the dataset

It has been observed that the ZINC dataset was used very often for the SMILE string representation. More specifically, a subset of the ZINC dataset that contains approximately 250,000 examples called ZINC250K. It was therefore decided to use the ZINC250K dataset for this work.



### Preprocessing

In preprocessing it was first of all chosen to choose the approach to transform the representation as a SMILE string. In order not to obtain excessively large sparse arrays with too much padding, after observing the distribution of the lengths of the strings, we have chosen to use only SMILE strings of length $\leq$ 60. In order to evaluate the goodness of the reconstruction, it was decided to split the main dataset in *train set* e *test set*: *train set* 70% - *test set* 30%. The training phase was carried out only in the train set.



### Models Building and Training

The chosen models type is the VAE (Varational Autoencoder). Two different VAEs have been implemented:

- Convolutional Encoder - Convolutional Decoder
- Convolutional Encoder - GRU (Multi-layer) + Linear layer

For both models, different training sessions were carried out with specific tuning set for 35 training periods. Having observed that in both models it is not necessary to have 35 epochs to obtain an efficient model, a new instance of each of the two model types with the most efficient setting out of 35 epochs has been trained for a smaller number of epochs resulting in truly efficient models.



### Molecules Reconstruction

In the reconstruction phase, the best model of both types was evaluated for its goodness in reconstructing the molecules. The reconstruction consists in obtaining the representation in latent space and then it is reconstructed obtaining the starting molecule. The more the model has learned latent space well, the better it will be able to return to the starting molecule. It was decided to evaluate even better it was decided to observe the accuracy of the molecule reconstruction and the validity ratio of the reconstructed molecules both on the test set and on the train set.



### Molecules Synthesis

In the synthesis phase, new molecules were generated by sampling from the prior distribution chosen for the latent space. To generate new molecules:

<p align = "center">
  <img src= "https://latex.codecogs.com/gif.latex?stdev&space;\times&space;\mathcal{N}(0,&space;1)&space;&plus;&space;latent" />

To generate new molecules, it was chosen to start from a molecule (i.e. from the representation of the latent space of the chosen molecule) and perturb it through the above formula in order to obtain new molecules. The only unknown in the formula is the standard deviation *stdev*. To observe the generation as we move further and further away from the starting molecule it was decided to use the following values for *stdev*: [0.045, 0.065, 0.085]. So it was decided to make 5000 samples for each of the three chosen values.

Furthermore, it was decided to choose two different starting molecules to generate new molecules not present in the train set. For each of these, 5000 samples were made for each possible value of the standard deviation.

Given ![](https://latex.codecogs.com/gif.latex?\mathcal{G}) the set of chemically valid molecules, ![](https://latex.codecogs.com/gif.latex?\mathcal{D}) the train set, ![](https://latex.codecogs.com/gif.latex?n) the number of syntatically valid generated molecules ![](https://latex.codecogs.com/gif.latex?n_{samp}), the following metrics were used to evaluate the generation of molecules:

- *Ratio Syntactically Validity*: ![](https://latex.codecogs.com/gif.latex?\frac{n}{n_{samp}})
- *Ratio Chemical Validity*: ![](https://latex.codecogs.com/gif.latex?\frac{|\mathcal{G}|}{n})
- ![](https://latex.codecogs.com/gif.latex?Uniqness:&space;\frac{|set(\mathcal{G})|}{n})
- *Novelty*: ![](https://latex.codecogs.com/gif.latex?1&space;-&space;\frac{|\mathcal{G}&space;\cap&space;\mathcal{D}|}{|\mathcal{G}|})
- *Similarity Ratio*: ![](https://latex.codecogs.com/gif.latex?\frac{\sum_{i=0}^{|\mathcal{G}|}&space;\sum_{j=i&plus;1}^{|\mathcal{G}|}&space;TanimotoSim(\mathcal{G}_i,&space;\mathcal{G}_j)}{\frac{|\mathcal{G}|(|\mathcal{G}|-&space;1)}{2}})

where TanimotoSim is the Tanimoto Similarity defined as:


<p align = "center">
  <img src= "https://latex.codecogs.com/gif.latex?TanimotoSim(A,&space;B)&space;=&space;\frac{A&space;\cdot&space;B}{||A||^2&space;&plus;&space;||B||^2&space;-&space;A&space;\cdot&space;B}" />






## Description and Requirements

Python 3 was used for the development of the system. The system was developed and tested on 64-bit Python 3.6 and 64-bit Python 3.7 on Linux Systems. It has been partially and locally developed and tested on Linux distros: Linux Mint and Ubuntu. However, the main tasks of the work were developed and carried out on Google Colab. The core of the project is in the jupyter-notebook, so there are no specific scripts for evaluation, train, synthesis and so on.

As future developments it is planned to develop specific scripts that allow these computations locally or on machines other than those of Google Colab.

The libraries used:

- NumPy
- PyTorch (torch and torchvision)
- Scikit-Learn
- Pandas
- Tqdm
- Matplotlib
- Plotly
- RDKit

RDKit is a library for Chioinformatics and Machine Learning widely used in the current state of the art.


## References

<a id="1">[1]</a>
Elton, Daniel & Boukouvalas, Zois & Fuge, Mark & Chung, Peter. (2019). Deep learning for molecular design - a review of the state of the art. Molecular Systems Design & Engineering. 10.1039/C9ME00039A. 


## License

The license for this software is: GPLv3.
