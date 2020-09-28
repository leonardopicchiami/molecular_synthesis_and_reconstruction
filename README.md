# MOLECULAR SYNTHESIS & RECONSTRUCTION

This is the Python repository about the Deep Learning And Applied Artificial Intelligence Project 2019/2020 in the Computer Science Master's Degree course of La Sapienza University of Rome.


## Goal Description

The goal was to develop Deep Learning models that can reconstruct molecules as best as possible and generate new molecules. The idea was to analyze the paper [[1]](#1): choose a representation of the molecules among the proposed one, choose a type of model and try different models of the same type (RNN, VAE, AAE, GAN ..) and choose different metrics on which evaluate models. The actual goal has been to get models that reconstruct and generate new ones in a good enough way compared to other models that are state of the art.

The entire work is reported and performed using Google Colab in the notebook below.

**Title** | **Notebook** | **Notebook on Colab**
|------------ | ------------- | ------------ |
Molecular Syntesis & Reconstruction | [notebook](.) | [![Open in Colab](.)] |

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

In preprocessing it was first of all chosen to choose the approach to transform the representation as a SMILE string. The idea was to use one-hot-encoding in order to obtain a sparse binary matrix that encodes a given SMILE. In order not to obtain excessively large sparse matrices with too much padding, after observing the distribution of the lengths of the strings, 


