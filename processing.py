# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    processing.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:10:40 by ebennace          #+#    #+#              #
#    Updated: 2022/11/10 15:00:45 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf
import numpy as np

from numpy import ndarray
from tensorflow import Tensor
from keras.applications.vgg19 import preprocess_input
# ======================================== #

def create_batch_image(img):

    img = tf.expand_dims(tf.constant(img),axis=0)
    return (img)

# ======================================== #

def avoid_batch_dimensions(array):
    
    array = np.reshape(array, (array.shape[1], array.shape[2], array.shape[3]))
    return (array)

# ======================================== #

def preprocessing_img(img):
    
    img = inverse_normalize_image(img)
    preprocessed_img = preprocess_input(img)
    return preprocessed_img

# ======================================== #

def Normalize_image(img):
    img = img / 255.
    return (img)

# ======================================== #

def inverse_normalize_image(img):
    img  = img * 255
    return (img)

# ======================================== #