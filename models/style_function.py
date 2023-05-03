# *************************************************************************** #
#                                                                              #
#    style_function.py                                                         #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/15 13:38:04 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt 


from IPython.display import clear_output, display
from .functions.image import tensor_to_image

from .functions.compute import compute_and_optimize_gradient

from tensorflow.keras.applications import VGG19
from tensorflow.keras.optimizers import Optimizer
from tensorflow import Tensor
from tensorflow import Variable
from tensorflow import GradientTape 
from keras import Model

from .functions.processing import preprocessing_img
from .functions.vgg import get_features_map

from .functions.compute import gram_matrix
from .functions.image import clip_pixel

# ===================================================== # 
def display_pictures(style_img : Tensor, noise_img : Tensor):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].imshow(style_img)
    ax[0].title.set_text(f'Style Image {style_img.shape}')
    ax[0].axis('off')

    ax[1].imshow(noise_img)
    ax[1].title.set_text(f'Noise Image {noise_img.shape}')
    ax[1].axis('off')
    
# ===================================================== # 

def display_generated_img(generated_img : Variable):
    
    clear_output(wait=True)
    display(tensor_to_image(generated_img))
    
# ===================================================== # 

def create_list_of_vgg_layer():

    style_layer_names   = ['block1_conv1',
                           'block2_conv1',
                           'block3_conv1',
                           'block4_conv1',
                           'block5_conv1']

    return (style_layer_names)

# ===================================================== # 

def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

# ===================================================== # 

def create_multi_output_model(style_layers : list)-> Model:

    vgg19 = load_vgg19()
    
    layers_name = style_layers
    layers_output = list()
    
    for name in layers_name:
        layer = vgg19.get_layer(name)
        output = layer.output
        layers_output.append(output)

    multi_output_model = Model([vgg19.input], layers_output)
    multi_output_model.trainable = False

    return (multi_output_model)

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

def create_batch_image(img : Tensor):

    img = tf.expand_dims(tf.constant(img),axis=0)
    return (img)

# ===================================================== # 

def init_generated_img(style_img : Tensor):

    
    generated_img = create_noisy_imag(style_img)
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

def init_style_target(model : Model, style_img : Tensor):

    style_img = create_batch_image(style_img)
    features_map = get_features_map(model, style_img)
    style_target = extract_style(features_map)

    return (style_target)

# ===================================================== # 

def display_generated_img(generated_img : Variable):
    
    clear_output(wait=True)
    display(tensor_to_image(generated_img))

# ===================================================== # 
   
def extract_style(features_map):

    Grams_styles = list()
    
    for style in features_map:
        Gram = gram_matrix(style)
        Grams_styles.append(Gram)
    return Grams_styles

# ===================================================== # 

def compute_style_loss(style_generated : Tensor, 
                       style_target : Tensor):

    all_style_loss = list()

    for generated, target in zip(style_generated, style_target):

        style_layer_loss = tf.reduce_mean((generated - target)**2)
        all_style_loss.append(style_layer_loss)

    num_style_layers = len(all_style_loss)
    style_loss = tf.add_n(all_style_loss) / num_style_layers

    return (style_loss)

# ===================================================== # 

@tf.function
def update_style(model : Model,
                 style_target : Tensor, 
                 generated_img : Variable,
                 optimizer : Optimizer):

    with GradientTape() as tape :

        features_map = get_features_map(model, generated_img)
        style_generated = extract_style(features_map)
        
        loss = compute_style_loss(style_generated, style_target)

    generated_img = compute_and_optimize_gradient(tape,
                                                  optimizer, 
                                                  generated_img,
                                                  loss)

# ===================================================== # 