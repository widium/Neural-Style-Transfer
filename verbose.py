# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    verbose.py                                         :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 09:00:04 by ebennace          #+#    #+#              #
#    Updated: 2022/11/10 21:35:00 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import numpy as np
import matplotlib.pyplot as plt 

from numpy import ndarray
from tensorflow import Variable, Tensor
from image import tensor_to_image

from IPython.display import clear_output, display
# ======================================== #

def print_features_maps_style(style_target):
    print("Style : ")
    for name, output in style_target.items():
        print(" ==", name, " ==")
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())

# ======================================== #

def print_features_maps_content(content_target):
    print("Content : ")
    for name, output in content_target.items():
        print(" ==", name, " ==")
        print("  shape: ", output.numpy().shape)
        print("  min: ", output.numpy().min())
        print("  max: ", output.numpy().max())
        print("  mean: ", output.numpy().mean())

# ======================================== #

def display_pictures(content_img : Tensor, style_img : Tensor):

    fig, ax = plt.subplots(1, 2, figsize=(20, 5))

    ax[0].imshow(content_img)
    ax[0].title.set_text(f'Content Image {content_img.shape}')
    ax[0].axis('off')

    ax[1].imshow(style_img)
    ax[1].title.set_text(f'Style Image {style_img.shape}')
    ax[1].axis('off')
    
# ======================================== #
    
def display_convertion_style(content_img : Tensor, style_img : Tensor, generated_img : Variable):
    # displaying content, style and generated style transferred images
    fig, ax = plt.subplots(1, 3, figsize=(15,10))

    ax[0].imshow(content_img)
    ax[0].title.set_text('Content Image')
    ax[0].axis('off')

    ax[1].imshow(style_img)
    ax[1].title.set_text('Style Image')
    ax[1].axis('off')

    generated_img = np.array(tensor_to_image(generated_img))
    ax[2].imshow(generated_img)
    ax[2].title.set_text('Generated Image')
    ax[2].axis('off')

    plt.show()

# ======================================== #

def display_generated_img(generated_img : Variable,
                          epoch : int,
                          num_epochs : int):
    
    
    if (epoch % 50 == 0):
        clear_output(wait=True)
        display(tensor_to_image(generated_img))
    
    print(f"epoch : {epoch}")
    print(f'Progression : {epoch*100/num_epochs}%')