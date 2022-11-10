# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    loss.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:09:12 by ebennace          #+#    #+#              #
#    Updated: 2022/11/10 19:39:26 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf

from tensorflow import Tensor, Variable
# ======================================== #

def compute_content_loss(content_generated, 
                         content_target):
    
    content_loss = tf.reduce_mean((content_generated - content_target)**2)
    return (content_loss)

# ======================================== #

def compute_style_loss(style_generated, 
                       style_target):

    all_style_loss = list()

    for generated, target in zip(style_generated, style_target):

        style_layer_loss = tf.reduce_mean((generated - target)**2)
        all_style_loss.append(style_layer_loss)

    num_style_layers = len(all_style_loss)
    style_loss = tf.add_n(all_style_loss) / num_style_layers

    return (style_loss)

# ======================================== #

def compute_total_loss(style_generated,
                       content_generated,
                       style_target,
                       content_target,
                       style_weight,
                       content_weight):

        content_loss = compute_content_loss(content_generated, content_target)
        style_loss = compute_style_loss(style_generated, style_target)

        total_loss = (style_weight * style_loss) + (content_weight * content_loss)

        return (total_loss)
    
# ======================================== #