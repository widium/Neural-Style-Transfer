# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    compute.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:05:10 by ebennace          #+#    #+#              #
#    Updated: 2022/11/21 11:48:50 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf

from tensorflow import Tensor
from tensorflow import GradientTape
from tensorflow.keras.optimizers import Optimizer
from tensorflow import Variable

from function.image import clip_pixel

# ======================================== #
# def gram_matrix(input_tensor : Tensor):

#   Gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
#   input_shape   = tf.shape(input_tensor)
#   num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) #
#   Gram_Normalized = Gram/num_locations
#   return (Gram_Normalized)

# ======================================== #

def flatten_filters(Feature_Maps):
    
    batch = int(Feature_Maps.shape[0])
    nbr_pixels = int(Feature_Maps.shape[1] * Feature_Maps.shape[2])
    nbr_filter = int(Feature_Maps.shape[3])
    
    matrix_pixels = tf.reshape(Feature_Maps, (batch, nbr_pixels, nbr_filter))
    return (matrix_pixels)

# ======================================== #

def normalize_matrix(G, Feature_Maps):

    height =  tf.cast(Feature_Maps.shape[1], tf.float32)
    width =  tf.cast(Feature_Maps.shape[2], tf.float32)
    number_pixels = height * width
    G = G / number_pixels
    return (G)

# ======================================== #

def gram_matrix(Feature_Maps):

    F = flatten_filters(Feature_Maps)
    Gram = tf.matmul(F, F, transpose_a=True)
    Gram = normalize_matrix(Gram, Feature_Maps)
    return Gram

# ======================================== #

def compute_and_optimize_gradient(tape : GradientTape, 
                                  optimizer : Optimizer, 
                                  generated_img : Variable, 
                                  loss : float):

    gradient = tape.gradient(loss, generated_img)
    optimizer.apply_gradients([(gradient, generated_img)])
    generated_img.assign(clip_pixel(generated_img))

    return (generated_img)

# ======================================== #