###############################################################################################################
#                                                                                                             #
# FileName    [one_hot_encoding.py]                                                                           #
#                                                                                                             #
# PackageName [preprocessing]                                                                                 #
#                                                                                                             #
# Synopsis    [This file contains the class that manages one hot encoding, performs both one hot encoding     #
#              for a SMILE list matrix and a single smile given as input to a method as a static method.      #
#              The dictionary for one hot encoding is given as input to the constructor,                      #
#              however, the charset globally defined in this file is taken by default..]                      #
#                                                                                                             #
# Author      [Leonardo Picchiami]                                                                            #
#                                                                                                             #
###############################################################################################################



import numpy as np
from typing import List, Union


CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
        '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
        'c', 'l', 'n', 'o', 'r', 's']


class OneHotEcodingHandler(object):
    '''
    Class that manages the one hot encoding, performs both the one hot encoding for a SMILE list matrix and 
    a single smile given in input to a method as a static method. The dictionary for one hot encoding is given 
    as input to the constructor, however, the charset globally defined in this file is taken by default.

    Attributes
    ----------
    smiles : list
        List of SMILE strings on which to perform one hot encoding.

    charset : list
        It is a string list where each string represents a character of the vocabulary needed for one hot encoding. 

    pad_length : int
        Fixed length of each string for encoding.
    '''

    
    def __init__(self, smiles : List = None , charset : List = CHARSET, pad_length : int = 32) -> None:
        '''
        Parameters
        ----------e
        smiles : list, default None
            List of SMILE strings on which to perform one hot encoding.

        charset : list, default global CHARSET
            It is a string list where each string represents a character of the vocabulary needed for one hot encoding.

        pad_length : int, default 32
            Fixed length of each string for encoding.
        '''

        self.__smiles = smiles
        self.__charset = CHARSET
        self.__pad_length = pad_length


        
    def encode_all_smiles(self) -> Union[np.ndarray, None]:
        '''
        Performs one hot encoding of each smile in the constructor input data.

        Returns
        -------
        one_hot_encoding : np.ndarray or None
            Matrix that contains the one hot matrix of each SMILES data.
        '''
        
        if self.__smiles:
            return np.array([self.one_hot_encode(smi) for smi in self.__smiles])
        else:
            return None

        
        
    def one_hot_encoding(self, smile : str) -> np.ndarray:
        '''
        Performs one hot encoding of the smile given in input.

        Parameters
        ----------
        smile : str
            SMILE input string on which to encode.

        Returns
        -------
        one-hot-encoding : np.ndarray
            Numpy array that contains the one hot encoding of the input smile.
        '''
        
        one_hot_matrix = []
        padded_smiles = self.smile_padding(smile)
        
        for c in padded_smiles:
            index = self.__charset.index(c)
            one_hot_matrix.append(
                [int(x) for x in [ix == index for ix in range(len(self.__charset))]]
            )

        return np.array(one_hot_matrix)


    
    def smile_padding(self, smile : str) -> str:
        '''
        Adds right padding to the SMILE given as input.

        Returns SMILE with padding if pad_length is greater than smile length, otherwise return original smile.

        Parameters
        ----------
        smile : str
            SMILE string on which to add padding.

        Returns
        -------
        smile : str
            String or containing only the original smiley or the original smiley with padding (empty spaces).
        
        '''
        
        if self.__pad_length - len(smile) > 0: 
            return smile + " " * (self.__pad_length - len(smile))
        else:
            return smile
