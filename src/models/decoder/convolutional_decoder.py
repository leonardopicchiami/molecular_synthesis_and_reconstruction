###############################################################################################################
#                                                                                                             #
# FileName    [convolutional_decoder.py]                                                                      #
#                                                                                                             #
# PackageName [models.decoder]                                                                                #
#                                                                                                             #
# Synopsis    [This file contains the class that models the convolutional neural network that plays           #
#              the role of decoder. It inherits the Pytorch nn.Module class by defining the various           #
#              layers in the constructor and overriding the forward method to define the actual behavior      #
#              of the network during training.]                                                               #
#                                                                                                             # 
# Author      [Leonardo Picchiami]                                                                            #
#                                                                                                             #
###############################################################################################################


import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    '''
    Class that models the convolutional neural network that plays the role of decoder. 
    It inherits the Pytorch nn.Module class by defining the various layers in the constructor 
    and overriding the forward method to define the actual behavior of the network during training.

    Attributes
    ----------
    one_hot_columns : int
        Column size of each one hot encoding matrix. 

    one_hot_rows : int
        Row size of each one hot encoding matrix.

    hidden_channels : int
        Reference hidden channel size of the convolutional layers

    fc : nn.Linear
        Linear layer that performs linear application from the latent space necessary for the autoencoder.

    conv1 : nn.ConvTranspose2d
        First layer that performs the 2D transposed convolution.

    conv2 : nn.ConvTranspose2d
        Second layer that performs the 2D transposed convolution.

    conv3 : nn.ConvTranspose2d
        Third layer that performs the 2D transposed convolution.

    activation : nn.ReLU or nn.SeLU
        Neural network activation function which can be ReLU or SeLU depending on what is chosen when the object is initialized.
    '''

    
    def __init__(self, in_channels : int, one_hot_columns : int,
                                          one_hot_rows : int,
                                          hidden_channels : int,
                                          latent_dim : int,
                                          kernel_size : int,
                                          stride : int,
                                          padding : int,
                                          activation_function : str) -> None:

        '''
        Parameters
        ----------
        in_channels : int
            Input channel size.

        one_hot_columns : int
            Column size of each one hot encoding matrix.
        
        one_hot_rows : int
            Row size of each one hot encoding matrix.

        hidden_channels : int
            Reference hidden channel size of the convolutional layers.

        latent_dim : int
            Latent space dimension size of autoencoder.
        
        kernel_size : int
            Square kernel size.

        stride : int
            Size of the stride in the convolution.

        padding : int
            Size of the padding around the matrix.

        activation_function : str
            String aimed at choosing whether to use the ReLU or SeLU activation function.
        '''

        super().__init__()

        self.__one_hot_columns = one_hot_columns
        self.__one_hot_rows = one_hot_rows
        self.__hidden_channels = hidden_channels


        #Linear layer from latent representation        
        self.__fc = nn.Linear(in_features = latent_dim,
                              out_features = hidden_channels*2*one_hot_columns*one_hot_rows)

        
        #Convolutional layers
        self.__conv1 = nn.ConvTranspose2d(in_channels = hidden_channels*2, 
                                          out_channels = hidden_channels, 
                                          kernel_size = kernel_size, 
                                          stride = stride, 
                                          padding = padding)
        
        self.__conv2 = nn.ConvTranspose2d(in_channels = hidden_channels, 
                                          out_channels = int(hidden_channels/2), 
                                          kernel_size = kernel_size, 
                                          stride = stride, 
                                          padding = padding)
        
        self.__conv3 = nn.ConvTranspose2d(in_channels = int(hidden_channels/2), 
                                          out_channels = in_channels, 
                                          kernel_size = kernel_size, 
                                          stride = stride, 
                                          padding = padding)
        

        #activation function
        if activation_function == "relu":
            self.__activation = nn.ReLU()
        elif activation_function == "selu":
            self.__activation = nn.SeLU()


            
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        '''
        Overriding the forward method which does:
            1. Application of the linear map of the linear layer.
            2. The application of the 2d transposed convolution as a linear map followed by the non-linear 
               application of the ReLU (or SeLU) function for the first layer.
            3. The application of the 2d transposed convolution as a linear map followed by the non-linear 
               application of the ReLU (or SeLU) function for the second layer.
            4. The application of the 2d trasposed convolution as a linear map followed by the non-linear
               application of the sigmoid fuction for the third layer returning the decoding as output.

        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor containing the latent space to be reconstructed.

        Returns
        -------
        x_out : torch.Tensor
            Output tensor containing the decoder otuput.
        '''
        
        x_out = self.__fc(x)
        x_out = x_out.contiguous().view(x_out.size(0), self.__hidden_channels*2, self.__one_hot_columns, self.__one_hot_rows)
        x_out = self.__activation(self.__conv1(x_out))
        x_out = self.__activation(self.__conv2(x_out))
        x_out = torch.sigmoid(self.__conv3(x_out))
        return x_out
