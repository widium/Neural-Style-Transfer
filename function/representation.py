# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    representation.py                                  :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 09:31:43 by ebennace          #+#    #+#              #
#    Updated: 2022/11/16 05:58:26 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from function.extract import get_styles_in_features_map
from function.processing import inverse_normalize_image

# =================================================== # 
    
def plot_features_map(features_map):
    # plot the output from each block
    height = 2
    width = 2

    for map in features_map:
        # plot all 64 maps in an 8x8 squares
        place = 1
        for _ in range(height):
            for _ in range(width):
                # specify subplot and turn of axis
                ax = plt.subplot(height, width, place)
                ax.set_xticks([])
                ax.set_yticks([])
                # plot filter channel in grayscale
                plt.imshow(map[0, :, :, place-1])
                place += 1
        # show the figure
        plt.show()

# =================================================== # 
      
def plot_features_map_style(features_map : list):
    
    style_outputs  = get_styles_in_features_map(features_map)
    nbr = len(style_outputs)
    fig, ax = plt.subplots(1, nbr, figsize=(20,12))

    i = 0
    for features in style_outputs :
           
        tensor = inverse_normalize_image(features)
        array = np.array(tensor, dtype=np.uint8)
        print("SHAPE : ", array.shape)
        array = np.reshape(array, (array.shape[1], array.shape[2]))
        ax[i].plot(array)
        ax[i].title.set_text(f"Shape {array.shape}")
        i += 1

    plt.show()

# =================================================== # 

def plot_style_representation(style_target : list):
    
    nbr = len(style_target)
    fig, ax = plt.subplots(1, nbr, figsize=(20,12))

    i = 0
    for gram in style_target :
           
        tensor = inverse_normalize_image(gram)
        array = np.array(tensor, dtype=np.uint8)
        array = np.reshape(array, (array.shape[1], array.shape[2]))
        sns.heatmap(array, ax=ax[i])
        i += 1

    plt.show()