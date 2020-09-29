################################################################################################################
#                                                                                                              #
# FileName    [one_hot_decoding.py]                                                                            #
#                                                                                                              #
# PackageName [preprocessing]                                                                                  #
#                                                                                                              #
# Synoposis   [This file contains the class that manages one hot decoding, performs one hot decoding           #
#              for both a matrix of one-hot-matrixes and a single one-hot-matrix given as input to a method as #
#              a static method. The dictionary for one hot encodingis given as input to the constructor,       #
#              however, the charset globally defined in this file is taken by default.]                        #
#                                                                                                              #
# Author      [Leonardo Picchiami]                                                                             #
#                                                                                                              #
################################################################################################################


import numpy as np
from typing import List, Union


CHARSET = [' ', '#', '(', ')', '+', '-', '/', '1', '2', '3', '4', '5', '6', '7',
        '8', '=', '@', 'B', 'C', 'F', 'H', 'I', 'N', 'O', 'P', 'S', '[', '\\', ']',
        'c', 'l', 'n', 'o', 'r', 's']



class OneHotDecodingHandler(object):
    '''
    Class that manages one hot decoding, performs one hot decoding for both a matrix of one-hot-matrixes 
    and a single one-hot-matrix given as input to a method as a static method. The dictionary for one hot encoding 
    is given as input to the constructor, however, the charset globally defined in this file is taken by default.

    Attributes
    ----------
    one_hot_matrix_list : np.ndarray
        Numpy array containing the one hot encoded matrixes of different SMILE strings.

    charset : list
        It is a string list where each string represents a character of the vocabulary needed for one hot encoding.
    '''

    
    def __init__(self, one_hot_matrix_list : np.ndarray = None, charset : List = CHARSET) -> None:
        '''
        Parameters
        ----------
        one_hot_matrix_list : np.ndarray
            Numpy array containing the one hot encoded matrixes of different SMILE strings.

        charset : list
            It is a string list where each string represents a character of the vocabulary needed for one hot encoding.
        '''

        self.__one_hot_matrix_list = one_hot_matrix_list
        self.__charset = charset


        
    def decode_all_smiles(self) -> List[str]:
        '''
        Decode all the one-hot-matrix given as input to the constructor.

        Returns
        -------
        smile_list : list
            List of all decoded SMILE strings.
        '''
        
        smiles_list = []
        if self.__one_hot_matrix:
            for k in range(len(self.__one_hot_matrix_list)):
                decoded_smile = self.one_hot_decoding(self.__one_hot_matrix_list[k])
                smile_list.append(decoded_smile)

        return smiles_list


    
    def one_hot_decoding(self, one_hot_matrix : np.ndarray) -> str:
        '''
        Decodes the previous one-hot-encoding by returning the SMILE string without padding, 
        then eliminating all right empty space.

        Parameters
        ----------
        one_hot_matrix : np.ndarray
            Numpy array represent the one-hot-encoding matrix of a fixed smile.
        
        Returns
        -------
        smile_string : str
            Decoded SMILE string without padding.
        '''
        
        decoded_smile = str()
        for i in range(len(one_hot_matrix)):
            index = np.argmax(one_hot_matrix[i])
            decoded_smile += self.__charset[index.item()]
        return self.delete_smile_padding(decoded_smile)


    
    def one_hot_decoding_with_padding(self, one_hot_matrix : np.ndarray) -> str:
        '''
        Decodes the previous one-hot-encoding by returning the SMILE string with padding, 
        then not eliminating all right empty space.

        Parameters
        ----------
        one_hot_matrix : np.ndarray
            Numpy array represent the one-hot-encoding matrix of a fixed smile.
        
        Returns
        -------
        decoded_smile : str
            Decoded SMILE string with padding.
        '''

        decoded_smile = str()
        for i in range(len(one_hot_matrix)):
            index = np.argmax(one_hot_matrix[i])            
            decoded_smile += self.__charset[index.item()]
        return decoded_smile


    
    def delete_smile_padding(self, decoded_smile : str) -> str:
        '''
        Deletes the smile padding string, i.e. it eliminates the spaces to the right of the SMILE characters.

        Returns the decoded SMILE string without right empty spaces if are present, otherwise it returns the original string.

        Parameters
        ----------
        decoded_smile : str
            SMILE string decoded with padding.

        Returns
        -------
        decoded_smile : str
            Substring of the original string that does not consider right empy spaces, or is the original string.
        '''
        
        if decoded_smile.find(' ') != -1:
            return decoded_smile[:decoded_smile.find(' ')]
        else:
            return decoded_smile

