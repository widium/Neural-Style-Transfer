# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    vgg.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 15:25:21 by ebennace          #+#    #+#              #
#    Updated: 2022/11/15 15:29:16 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf
import numpy as np

from tensorflow.keras.applications import VGG19
from keras import Model
from tensorflow import Tensor

from function.processing import preprocessing_img

# ======================================== #

def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

# ======================================== #

def create_list_of_vgg_layer():

    content_layer_name  = ['block5_conv2']

    style_layer_names   = ['block1_conv1',
                           'block2_conv1',
                           'block3_conv1',
                           'block4_conv1',
                           'block5_conv1']

    return (content_layer_name, style_layer_names)

# ======================================== #

def create_multi_output_model(style_layers : list, content_layers : list)-> Model:

    vgg19 = load_vgg19()
    
    layers_name = style_layers + content_layers
    layers_output = list()
    
    for name in layers_name:
        layer = vgg19.get_layer(name)
        output = layer.output
        layers_output.append(output)

    multi_output_model = Model([vgg19.input], layers_output)
    multi_output_model.trainable = False

    return (multi_output_model)

# ======================================== #

def get_features_map(model : Model, img : Tensor)->list:

        process_img = preprocessing_img(img)
        features_map = model(process_img)

        return (features_map)