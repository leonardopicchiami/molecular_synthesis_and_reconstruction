###################################################################################################################
#                                                                                                                 # 
# FileName    [smile_dataset.py]                                                                                  #
#                                                                                                                 #
# PackageName [preprocessing]                                                                                     #
#                                                                                                                 #
# Synopsis    [This file contains the class that models the custom dataset for model training.                    #
#              It inherits the Pytorch Dataset class to be able to return the one-hot-encoding representation     #
#              of the i-th SMILE at each iteration. If a transformation is given as input, the transformation     # 
#              is applied to the one-hot-encoding representation.]                                                #
#                                                                                                                 #
# Author      [Leonardo Picchiami]                                                                                #
#                                                                                                                 #
###################################################################################################################


from typing import Union

import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import numpy as np
import pandas as pd

import one_hot_encoding as enc


class SmileDataset(Dataset):
    '''
    Class that models the custom dataset for model training.It inherits the Pytorch Dataset class to be able 
    to return the one-hot-encoding representation of the i-th SMILE at each iteration. 
    If a transformation is given as input, the transformation is applied to the one-hot-encoding representation.
    '''

    def __init__(self, csv_file : str, smile_length : int, transform : Compose = None) -> None:
        
        self.__smiles_frame = pd.read_csv(csv_file)
        self.__transform = transform
        self.__smile_length = smile_length

        

    def __len__(self) -> int:
        '''
        Override of special method to define len function for this class.

        The len of SmileDataset is the len of pandas dataframe modeling the smile data.

        Returns
        -------
        len(smiles_frame) : int
        
        '''
        return len(self.__smiles_frame)

    
    
    def __getitem__(self, idx : Union[torch.Tensor, int]) -> Union[torch.Tensor, np.ndarray]:
        '''
        Override special method which returns the element in the idx position of the dataset. 
        The dataset is modeled internally by extracting the smile string in position ixd in the dataframe 
        and returning either the numpy array of the one hot encoding of the selected smile or, 
        if the transformation is active, the tensor of the same hot encoding.

        Parameters
        ----------
        idx : torch.Tensor or int
            Index of returning dataset element.


        Returns
        -------
        one_hot_encode(smile) : np.ndarray or torch.Tensor
            One hot encoding matrix of smile string.        
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        smile = self.__smiles_frame['smiles'][idx] 
        one_hot_enc = enc.OneHotEcodingHandler(pad_length=self.__smile_length)

        if self.__transform:
            return self.__transform(one_hot_enc.one_hot_encoding(smile)) 
        else:
            return one_hot_enc.one_hot_encoding(smile)


        
    def get_smile_string_element(self, idx : Union[torch.Tensor, int]) -> str:
        '''
        Returns the SMILE string at idx position in the dataset.

        Returns
        -------
        smile_string : str
            the SMILE string at idx position in the dataset.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.__smiles_frame['smiles'][idx]

    

    def get_max_smile_length(self) -> int:
        '''
        Getter method that returns the maximum length of the SMILE string lengths in the dataset.

        Returns
        -------
        max : int
            The maximum length of the SMILE string lengths in the dataset.
        '''
        
        return self.__smiles_frame['smiles'].str.len().max()


    
