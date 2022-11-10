# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    init.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:18:30 by ebennace          #+#    #+#              #
#    Updated: 2022/11/10 14:59:10 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf
import numpy as np

from numpy import ndarray
from tensorflow import Tensor

from tensorflow import Variable
from keras import Model

from Model import get_features_map
from processing import create_batch_image
from image import create_noisy_imag, clip_pixel
from extract import extract_content, extract_style
# ======================================== #

def init_content_target(model , content_img):

        content_img = create_batch_image(content_img)
        features_map = get_features_map(model, content_img)
        content_target = extract_content(features_map)

        return content_target

# ======================================== #

def init_style_target(model, style_img):

    style_img = create_batch_image(style_img)
    features_map = get_features_map(model, style_img)
    style_target = extract_style(features_map)

    return style_target

# ======================================== #

def init_generated_img(content_img, 
                       noise_ratio=0.35):

    
    generated_img = create_noisy_imag(content_img, noise_ratio)
    generated_img = clip_pixel(generated_img)
    generated_img = create_batch_image(generated_img)
    generated_img = Variable(generated_img)

    return (generated_img)