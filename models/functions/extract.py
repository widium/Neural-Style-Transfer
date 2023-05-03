# *************************************************************************** #
#                                                                              #
#    extract.py                                                                #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/10 09:21:02 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** ## =============== Import =================== #
import tensorflow as tf
import numpy as np

from numpy import ndarray
from tensorflow import Tensor

from tensorflow import Variable

from .compute import gram_matrix
# ======================================== #

def extract_content(features_map):

    content = get_content_in_features_map(features_map)
    return (content)

# ======================================== #

def extract_style(features_map):

    Grams_styles = list()
    style_outputs  = get_styles_in_features_map(features_map)
    
    for style in style_outputs:
        Gram = gram_matrix(style)
        Grams_styles.append(Gram)
    return Grams_styles

# ======================================== #

def get_styles_in_features_map(features_map)-> Tensor:
    
    style_outputs  = features_map[:-1]
    return (style_outputs)
    
# ======================================== #

def get_content_in_features_map(features_map)-> Tensor:
    
    content  = features_map[-1]
    return (content)

# ======================================== #