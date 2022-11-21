# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    content_model.py                                   :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 13:15:39 by ebennace          #+#    #+#              #
#    Updated: 2022/11/18 14:01:42 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
sys.path.append("..")

from tensorflow.keras.optimizers import Adam


import matplotlib.pyplot as plt 

from time import time

from function.image import get_picture_name
from function.image import load_image

from function.save import add_frame
from function.save import save_evolution

from .content_function import display_pictures
from .content_function import display_generated_img
from .content_function import create_model
from .content_function import init_content_target
from .content_function import init_generated_img
from .content_function import init_noise_image
from .content_function import update_content

# ===================================================== # 

class Model_Content_Representation:
    
    # ===================================================== # 
    def __init__(self, optimizer=Adam(learning_rate=0.02), style_weight=1e6, content_weight=5e0, noise_ratio=0.20):
        super().__init__()
        self.optimizer = optimizer
        self.content_layers = ['block4_conv4']
        self.model = create_model(self.content_layers)
        self.content_img = None
        self.generated_img = None
        self.noise_img = None
        self.frames = list()

    # ===================================================== # 
    
    def import_img(self, content_img):

        self.content_img = load_image(content_img)
        self.noise_img = init_noise_image(self.content_img)
        display_pictures(self.content_img, self.noise_img)

    # ===================================================== # 
    
    def recreate_content(self, num_epochs, create_gif=False):

        target_content = init_content_target(self.model, self.content_img)
        self.generated_img = init_generated_img(self.content_img)
        
        start = time()
        for epoch in range(num_epochs) :

            update_content(self.model,
                         target_content, 
                         self.generated_img, 
                         self.optimizer)

            display_generated_img(self.generated_img)
            if (create_gif == True):
                add_frame(self.frames, self.generated_img, epoch)
             
        end = time()
        print("Total training time: {:.1f} seconds".format(end-start))
        if (create_gif == True):
            save_evolution(self.frames, self.content_img, self.noise_img, self.generated_img)