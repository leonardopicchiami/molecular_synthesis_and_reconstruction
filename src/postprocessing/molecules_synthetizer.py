###############################################################################################################################
#                                                                                                                             #   
# FileName    [molecules_synthetizer.py]                                                                                      #
#                                                                                                                             #
# PackageName [postprocessing]                                                                                                #
#                                                                                                                             #
# Synopsis    [This file contains the class that manages the generation of molecules starting from a given molecule           #
#              represented as a SMILE string and generates a fixed number of molecules in accordance with the average         #
#              of the molecule's latent space and with the fixed standard deviation.                                          #
#              It also analyzes the number of syntactically valid molecules generated from the samples and inserts a          #
#              fixed number of molecules into a grid image.]                                                                  #
#                                                                                                                             #
# Author      [Leonardo Picchiami]                                                                                            #                     #                                                                                                                             #       
###############################################################################################################################



from typing import List

import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from rdkit import Chem
from rdkit.Chem import Draw
from PIL import Image

from one_hot_encoding import OneHotEcodingHandler
from one_hot_decoding import OneHotDecodingHandler
from molecular_vae import MolecularVae



class MoleculesSynthetizer(object):
    '''
    This file contains the class that manages the generation of molecules starting from a given molecule represented as a SMILE string and 
    generates a fixed number of molecules in accordance with the average of the molecule's latent space and with the fixed standard deviation.
    It also analyzes the number of syntactically valid molecules generated from the samples and inserts a fixed number of molecules into a grid image.
    

    Attributes
    ----------
    model : MolecularVae
        Trained model used for the molecules synthesis.

    smile_string : string
        SMILE string used as a starting molecule for generation.

    latent_dim : int
        MolecularVae's Latent space size.
    
    one_hot_encoder : OneHotEcodingHandler
        Handler that manages the one hot encoding of a SMILE or a list of SMILE.

    one_hot_decoder : OneHotDecodingHandler
        Handler that manages the one hot decoding of a one hot matrix or a list of one hot matrix.

    device : str, default cpu
        String that indicates the device on which the model operations are computed. (cpu or cuda)

    synthetized_samples : list
        List of samples from latent space and transformed into SMILE string representation.

    synthetized_synt_valid_molecules : list
        List of syntactically valid generated molecules transformed into the SMILE string representation.

    valid_molecules_ratio : int
        Ratio of syntactically valid molecules generated with respect to the total number of samples.
    '''

    
    def __init__(self, model : MolecularVae, smile_string : str, latent_dim : int,
                                                                 one_hot_encoder : OneHotEcodingHandler,
                                                                 one_hot_decoder : OneHotDecodingHandler,
                                                                 device : str ='cpu') -> None:
        '''
        Parameters
        ----------
        model : MolecularVae
            Trained model used for the molecules synthesis.

        smile_string : string
            SMILE string used as a starting molecule for generation.

        latent_dim : int
            MolecularVae's Latent space size.
    
        one_hot_encoder : OneHotEcodingHandler
            Handler that manages the one hot encoding of a SMILE or a list of SMILE.

        one_hot_decoder : OneHotDecodingHandler
            Handler that manages the one hot decoding of a one hot matrix or a list of one hot matrix.

        device : str, default cpu
            String that indicates the device on which the model operations are computed. (cpu or cuda)

        '''
        
        self.__model = model
        self.__smile_string = smile_string
        self.__latent_dim = latent_dim
        self.__one_hot_encoder = one_hot_encoder
        self.__one_hot_decoder = one_hot_decoder
        self.__device = device
        self.__synthetized_samples = []
        self.__synthetized_synt_valid_molecules = []
        self.__valid_molecules_ratio = 0

        

    def one_hot_encode_smile_reference(self) -> np.ndarray:
        '''
        One hot encoding of the SMILE string from which to start for the generation.

        Returns
        -------
        one_hot_encode : np.ndarray
            one hot encoding matrix of the smile string from which generation starts.
        '''
        
        encoded_molecule = self.__one_hot_encoder.one_hot_encoding(self.__smile_string)
        return encoded_molecule.reshape(1, 1, encoded_molecule.shape[0], encoded_molecule.shape[1])


    
    def one_hot_decode_samples(self, encoded_samples_matrixes : torch.Tensor) -> List[str]:
        '''
        Performs one hot decoding of all samples played.

        Parameters
        ----------
        encoded_samples_matrixes : torch.Tensor
            Tensor that contains the one hot matrix of the decoded samples.

        Returns
        -------
            List of SMILE string representations of the generated sample.
        '''
        
        decoded_samples = []
        for i in range(len(encoded_samples_matrixes)):
            decoded_samples.append(self.__one_hot_decoder.one_hot_decoding(encoded_samples_matrixes[i][0]))
        return decoded_samples
            

    
    def synthetize_molecule(self, std_dev : float, num_samples : int) -> None:
        '''
        Synthesizes the molecules obtaining the representation of the latent space of the starting string, 
        sampling num_samples examples according to the chosen std_dev deviation, decodes the sampled samples and 
        brings them into the SMILE string representation.

        Parameters
        ----------
        std_dev : float
            Standard deviation value for sampling from latent space.

        num_samples : int
            Number of latent space samples to be performed.
        '''
        
        self.__model.eval()

        with torch.no_grad():
            smile_one_hot_encoding = self.one_hot_encode_smile_reference()
            
            mu, logvar = self.__model.to(self.__device).encoder(torch.from_numpy(smile_one_hot_encoding).float())
            latent = self.__model.to(self.__device).reparametrization2(mu, logvar)

            encoded_synthetized_mols = std_dev * torch.randn(num_samples, self.__latent_dim) + latent
            encoded_samples = self.__model.to(self.__device).decoder(encoded_synthetized_mols)

            self.__synthetized_samples = self.one_hot_decode_samples(encoded_samples)   


            
    def synthetize_syntatically_valid_molecules(self, std_dev : float, num_samples : int) -> None:
        '''
        It synthesizes syntactically valid molecules. 
        Then it performs the synthesis of samples, returning the SMILE string representation of the generated samples. 
        It is then checked that each sample represents a syntactically valid molecule; if yes, it is added to the list.
        
        Finally, the fraction of syntactically valid molecules generated with respect to the total number of samples is calculated.
        
        Parameters
        ----------
        std_dev : float
            Standard deviation value for sampling from latent space.

        num_samples : int
            Number of latent space samples to be performed.
        '''
        
        self.synthetize_molecule(std_dev, num_samples)
        self.__synthetized_synt_valid_molecules = []
        
        for sample in self.__synthetized_samples:
            mol = Chem.MolFromSmiles(sample, sanitize=False)
            if mol:
                self.__synthetized_synt_valid_molecules.append(sample)

        self.__valid_molecules_ratio = len(self.__synthetized_synt_valid_molecules) / len(self.__synthetized_samples)

        

    def smile_to_csv(self, save_path : str) -> None:
        '''
        Save valid molecules generated in a pandas dataframe in the path given in input.

        Parameters
        ----------
        save_path : str
            Path string where to save the smile dataframe.
        '''
        
        smile_frame = pd.DataFrame(self.__synthetized_synt_valid_molecules, columns = ['synthetized_smiles'])
        smile_frame.to_csv(save_path)
        

        
    def get_synthetized_syntactically_valid_unique_molecules(self) -> List[str]:
        '''
        Returns the list of unique molecules synthesized.

        Returns
        -------
        list(set(synthetized_valid_molecules)) : list
            List of molecules generated without repetitions.
        '''
        
        return list(set(self.__synthetized_synt_valid_molecules))


    
    def get_synthetized_syntactically_valid_molecules(self) -> List[str]:
        '''
        Returns the list of syntactically valid generated molecules.

        Returns
        -------
        synthetized_valid_molecules : list
            List of syntactically valid molecules generated.
        '''
        
        return self.__synthetized_synt_valid_molecules

    
    
    def get_synthetized_samples(self) -> List[str]:
        '''
        Returns the samples sampled, decoded and transformed into the SMILE string representation.

        Returns
        -------
        synthetized_samples : list
        '''
        
        return self.__synthetized_samples

    

    def get_syntactically_valid_molecules_ratio(self) -> float:
        '''
        Getter method that returns the ratio of valid synthetized molecules.

        Returns
        -------
        valid_molecules_ratio : float
            Ratio of syntactically valid synthetized molecules.
        '''
        
        return self.__valid_molecules_ratio

    
    
    def synthetized_molecules_grid(self, start_index : int, end_index : int, mols_per_rows : int,
                                                                             image_size : (int, int),
                                                                             save_path : str = None) -> Image:
        '''
        It takes a portion of the syntactically valid molecules, verifies that they are chemically valid 
        and inserts them into an image grid. If the path is given, it is saved in the specified path.

        Parameters
        ----------
        start_index : int
            Index from which to start selecting the generated molecules.

        end_index : int
            Index indicating the end of the range of molecules to be selected for the grid.

        mols_per_rows : int
            Number of molecules per row in the grid.

        image_size : tuple
            Size of the 2D grid image.

        save_path : str, default None
            Path to save the grid.


        Returns
        -------
        img :
            Grid image of synthesized molecules.
        '''

        synthetized_mols = []

        valid_uniques = self.get_synthetized_syntactically_valid_unique_molecules()
        for smile in valid_uniques:
            mol = Chem.MolFromSmiles(smile, sanitize=False)

            try:
                Chem.SanitizeMol(mol)
                synthetized_mols.append(mol)
            except:
                continue
            
        if end_index < len(synthetized_mols):
            end = len(synthetized_mols)
        else:
            end = end_index
            
        img = Draw.MolsToGridImage(
                synthetized_mols[start_index:end],
                molsPerRow = mols_per_rows,
                subImgSize = (image_size, image_size))

        if save_path:
            img.save(save_path)

        return img
        



    
