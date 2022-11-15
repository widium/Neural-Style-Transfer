# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    style_model.py                                      :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 13:08:56 by ebennace          #+#    #+#              #
#    Updated: 2022/11/15 13:08:57 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
sys.path.append("..")

import matplotlib.pyplot as plt 

from time import time
from tensorflow.keras.optimizers import Adam

from function.save import add_frame
from function.save import save_convertion

from function.image import load_image

from .style_function import create_list_of_vgg_layer
from .style_function import create_multi_output_model
from .style_function import init_style_target
from .style_function import init_generated_img
from .style_function import init_noise_image
from .style_function import update_style
from .style_function import display_generated_img
from .style_function import display_pictures

# ===================================================== # 

class Model_Style_Representation:

    # =========================== # 
    def __init__(self, optimizer=Adam(learning_rate=0.02)):
        super().__init__()
    
        self.optimizer = optimizer
        
        self.style_layers = create_list_of_vgg_layer()
        self.num_style_layers  = len(self.style_layers)
        
        self.model = create_multi_output_model(self.style_layers)
        
        self.style_img = None
        self.generated_img = None
        
        self.frames = list()

    # ============================== # 
    
    def import_img(self, style_img):

        self.style_img = load_image(style_img)
        noise_img = init_noise_image(self.style_img)
        display_pictures(self.style_img, noise_img)

    # ============================== # 
    
    def recreate_style(self, num_epochs):

        target_style = init_style_target(self.model, self.style_img)
        self.generated_img = init_generated_img(self.style_img)
            
        start = time()
        for epoch in range(num_epochs) :

            update_style(self.model,
                         target_style, 
                         self.generated_img, 
                         self.optimizer)

            display_generated_img(self.generated_img)
             
        end = time()
        print("Total training time: {:.1f} seconds".format(end-start))
        return (self.generated_img)

# ===================================================== # 