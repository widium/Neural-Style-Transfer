# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    style.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:53:35 by ebennace          #+#    #+#              #
#    Updated: 2022/11/10 19:54:35 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf
import numpy as np

from numpy import ndarray
from tensorflow import Tensor
from keras import Model

from tensorflow import Variable
from tensorflow.keras.optimizers import Optimizer

from Model import get_features_map
from loss import compute_total_loss
from compute import compute_and_optimize_gradient
from extract import extract_content, extract_style

# ======================================== #
@tf.function
def update_style(model,
                 style_target, 
                 content_target, 
                 generated_img,
                 style_weight,
                 content_weight, 
                 optimizer):

    with tf.GradientTape() as tape :

        features_map = get_features_map(model, generated_img)
        style_generated = extract_style(features_map)
        content_generated = extract_content(features_map)
        
        loss = compute_total_loss(style_generated,
                                  content_generated,
                                  style_target,
                                  content_target,
                                  style_weight,
                                  content_weight)

    generated_img = compute_and_optimize_gradient(tape,
                                                  optimizer, 
                                                  generated_img,
                                                  loss)
    
# ======================================== #