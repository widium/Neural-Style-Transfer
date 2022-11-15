# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    compute.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:05:10 by ebennace          #+#    #+#              #
#    Updated: 2022/11/15 15:28:38 by ebennace         ###   ########.fr        #
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

def gram_matrix(input_tensor : Tensor):

  Gram = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape   = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32) #

  Gram_Normalized = Gram/num_locations
  return (Gram_Normalized)

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