##############################################################################################################
#                                                                                                            # 
# FileName    [molecular_vae.py]                                                                             #
#                                                                                                            #
# PackageName [models]                                                                                       #
#                                                                                                            #
# Synopsis    [This file contains the class that models a generic molecular autoencoder that takes           #
#              an encoder and a decoder as input and performs the encoding and decoding operation during     #
#              the training using the reparameterization trick useful for the training backpropagation.]     #
#                                                                                                            #
# Author      [Leonardo Picchiami]                                                                           #
#                                                                                                            #
##############################################################################################################


import torch
import torch.nn as nn


class MolecularVae(nn.Module):
    
    '''
    Class that models a generic molecular autoencoder that takes an encoder and a decoder as input 
    and performs the encoding and decoding operation during the training using the reparameterization trick
    useful for the training backpropagation.

    In Pytorch the custom implementation of neural networks is carried out by selling the class nn.Module.

    Attributes
    ----------
    encoder : Encoder
        Encoder to perform the encoding task for the varational autoencoder. 
        The encoder will only be of the convolutional type.

    decoder : Decoder
        Decoder to perform the decoding task for the varational autoencoder.
        The encoder will be either convolutional or recurrent (GRU + Linear).

    repar_type : str
        String that allows you to select one of the two present implementations of the reparametrization trick.
    '''

    
    def __init__(self, encoder : nn.Module, decoder : nn.Module, repar_type : str) -> None:
        '''
        Autoencoder constructor.

        Parameters
        ----------
        encoder : nn.Module
            Encoder type to perform the encoding part of varational autoencoder's task

        decoder : nn.Module
            Decoder type to perform the decoding part of varational autoencoder's task

        repar_type : str
            String to select the type of implementation of the proposed reparametrization trick.
        '''
        
        super().__init__()
        self.__encoder = encoder
        self.__decoder = decoder
        self.__repar_type = repar_type


        
    def encoder(self, x : torch.Tensor) -> (torch.Tensor, torch.Tensor):
        '''
        Method that performs the econding task according to the type of input encoder to the constructor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the input examples to be encoded in the latent space

        Returns
        -------
        encoder(x) : tuple of torch.Tensor
        '''
        
        return self.__encoder(x)


        
    def decoder(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Method that performs the decoding task according to the type of input decoder to the constructor.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the latent space to be reconstructed

        Returns
        -------
        decoder(x) : torch.Tensor
            Output tensor containing the decoder otuput (input reconstruction)
        '''

        return self.__decoder(x)


        
    def reparametrization1(self, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:
        '''
        Method that implements the first version of the reparametrization trick which provides latent space 
        for a given input example.

        Parameters
        ----------
        mu : torch.Tensor
            Tensor containing the mean for the normal distribution according to the size of the latent space chosen.

        logvar : torch.Tensor
            Tensor containing the log-variance for the normal distribution according to the size of latent space chosen.
        

        Returns
        -------
        latent_mu : torch.Tensor
            The latent space re-sampled mean for normal distribution with fixed mean and standard deviation.

        mu : torch.Tensor
            The latent space mean previus obtained.
        '''
        
        if self.training:
            std = (logvar * 0.5).exp()
            return torch.distributions.Normal(loc=mu, scale=std).rsample()
        else:
            return mu

        
        
    def reparametrization2(self, mu : torch.Tensor, logvar : torch.Tensor) -> torch.Tensor:
        '''
        Method that implements the second version of the reparametrization trick which provides latent space 
        for a given input example.

        Parameters
        ----------
        mu : torch.Tensor
            Tensor containing the mean for the normal distribution according to the size of the latent space chosen.

        logvar : torch.Tensor
            Tensor containing the log-variance for the normal distribution according to the size of latent space chosen.
        

        Returns
        -------
        latent_mu : torch.Tensor
            the latent space re-sampled mean for normal distribution with fixed mean and standard deviation.

        mu : torch.Tensor
            the latent space mean previus obtained.
        '''
      
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = 1e-2 * torch.randn_like(std)
            w = eps.mul(std).add_(mu)
            return w
        else:
            return mu


        
    def forward(self, x : torch.Tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor):
        '''
        Inherited method that iterates the autoencoder task:
            - encoding
            - reparametrization (only in case of neural network training) 
            . decoding

        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing batch examples to process it.


        Returns
        -------
        (x_recon, latent_mu, latent_logvar) : tuple of torch.Tensor
            Output tuple of torch.Tensor containing: 
            - x_recon :  the reconstructed input batch examples x.
            - latent_mu : the latent mean obtained from encoder.
            - latent_logvar : the latent log-variance obtained from encoder.
        '''
        
        latent_mu, latent_logvar = self.encoder(x)

        if self.__repar_type == 'type1':
            latent = self.reparametrization1(latent_mu, latent_logvar)
        else:
            latent = self.reparametrization2(latent_mu, latent_logvar)

        x_recon = self.decoder(latent)
        return (x_recon, latent_mu, latent_logvar)
    
