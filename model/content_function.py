# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    content_function.py                                :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 13:16:06 by ebennace          #+#    #+#              #
#    Updated: 2022/11/17 07:18:23 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
sys.path.append("..")

# importing required libraries
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from IPython.display import clear_output
from IPython.display import display

from tensorflow import Variable, Tensor
from tensorflow import GradientTape 
from tensorflow.keras.optimizers import Optimizer
from tensorflow.keras.applications import VGG19

from keras import Model

from function.processing import preprocessing_img
from function.image import tensor_to_image
from function.compute import compute_and_optimize_gradient

# ===================================================== # 
def display_pictures(content_img : Tensor, noise_img : Tensor):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].imshow(content_img)
    ax[0].title.set_text(f'Content Image {content_img.shape}')
    ax[0].axis('off')

    ax[1].imshow(noise_img)
    ax[1].title.set_text(f'Noise Image {noise_img.shape}')
    ax[1].axis('off')
    
# ===================================================== # 

def display_generated_img(generated_img : Variable):
    
    clear_output(wait=True)
    display(tensor_to_image(generated_img))
    
# ===================================================== # 

def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

# ===================================================== # 

def create_model(content_layers : list)-> Model:

    vgg19 = load_vgg19()
    name = content_layers[0]
    layer = vgg19.get_layer(name)
    output = layer.output

    model = Model([vgg19.input], output)
    model.trainable = False

    return (model)

# ===================================================== # 

def get_features_map(model : Model, img : Tensor)->list:

        process_img = preprocessing_img(img)
        features_map = model(process_img)

        return (features_map)

# ===================================================== # 
 
def create_noisy_imag(img : Tensor):
    
    noise_filter = np.random.randn(*img.shape)
    return (noise_filter)

# ===================================================== # 

def clip_pixel(image : Tensor):
  
    cliped_imag = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
    return (cliped_imag)

# ===================================================== # 

def create_batch_image(img : Tensor):

    img = tf.expand_dims(tf.constant(img),axis=0)
    return (img)

# ===================================================== # 

def extract_content(features_map):

    content = features_map[0]
    return (content)

# ===================================================== # 

def init_content_target(model : Model, content_img : Tensor):

    content_img = create_batch_image(content_img)
    features_map = get_features_map(model, content_img)
    content_target = extract_content(features_map)

    return (content_target)

# ===================================================== # 

def init_generated_img(content_img : Tensor):

    generated_img = create_noisy_imag(content_img)
    generated_img = clip_pixel(generated_img)
    generated_img = create_batch_image(generated_img)
    generated_img = Variable(generated_img)

    return (generated_img)


# ===================================================== # 

def init_noise_image(style_img : Tensor):
    
    generated_img = create_noisy_imag(style_img)
    generated_img = clip_pixel(generated_img)
    generated_img = Variable(generated_img)
    
    return (generated_img)

# ===================================================== # 

def compute_content_loss(content_generated : Tensor, 
                         content_target : Tensor):
    
    content_loss = tf.reduce_mean((content_generated - content_target)**2)
    return (content_loss)

# ===================================================== # 

@tf.function
def update_content(model : Model,
                 content_target : Tensor, 
                 generated_img : Variable,
                 optimizer : Optimizer):

    with GradientTape() as tape :

        features_map = get_features_map(model, generated_img)
        content_generated = extract_content(features_map)
        
        loss = compute_content_loss(content_generated, content_target)

    generated_img = compute_and_optimize_gradient(tape,
                                                  optimizer, 
                                                  generated_img,
                                                  loss)

# ===================================================== # 