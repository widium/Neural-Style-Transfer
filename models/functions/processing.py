# *************************************************************************** #
#                                                                              #
#    processing.py                                                             #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/10 09:10:04 by  ebennace                                 #
#    Updated: 2023/05/04 11:37:55 by Widium                                    #
#                                                                              #
# **************************************************************************** ## =============== Import =================== #
import tensorflow as tf
import numpy as np

from numpy import ndarray
from tensorflow import Tensor
from keras.applications.vgg19 import preprocess_input
# ======================================== #

def create_batch_image(img : Tensor):

    img = tf.expand_dims(tf.constant(img),axis=0)
    return (img)

# ======================================== #

def remove_batch_dimension(array : ndarray):
    
    array = np.reshape(array, (array.shape[1], array.shape[2], array.shape[3]))
    return (array)

# ======================================== #

def preprocessing_img(img : Tensor):
    
    img = inverse_normalize_image(img)
    preprocessed_img = preprocess_input(img)
    return preprocessed_img

# ======================================== #

def Normalize_image(img : Tensor):
    img = img / 255.
    return (img)

# ======================================== #

def inverse_normalize_image(img : Tensor):
    img  = img * 255
    return (img)

# ======================================== #