# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    neural_style_transfert.py                          :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/16 06:02:53 by ebennace          #+#    #+#              #
#    Updated: 2022/11/16 06:08:59 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #


import sys
sys.path.append('..')

# ******************************************************* #

from time import time
from tensorflow.keras.optimizers import Adam

from function.init import init_style_target
from function.init import init_content_target
from function.init import init_generated_img

from function.vgg import create_list_of_vgg_layer
from function.vgg import load_vgg19
from function.vgg import create_multi_output_model

from function.image import load_image
from function.image import tensor_to_image

from function.verbose import display_pictures
from function.verbose import display_generated_img
from function.verbose import display_convertion_style

from function.save import add_frame
from function.save import save_convertion

from function.style import update_style

# ******************************************************* #
# tf.summary.scalar('batch_loss', loss)
# tf.summary.scalar('batch_mse', mse)


class Model_Style_Transfert:

    def __init__(self, optimizer=Adam(learning_rate=0.02), style_weight=1e6, content_weight=5e0, noise_ratio=0.20):
        super().__init__()
        self.style_weight = style_weight
        self.content_weight = content_weight
        self.noise_ratio = noise_ratio
        self.optimizer = optimizer
        self.content_layers, self.style_layers, = create_list_of_vgg_layer()
        self.num_style_layers  = len(self.style_layers)
        self.model = create_multi_output_model(self.style_layers, self.content_layers)
        self.style_img = None
        self.content_img = None
        self.generated_img = None
        self.frames = list()

    def import_img(self, content_img, style_img):

        self.content_img = load_image(content_img)
        self.style_img = load_image(style_img)
        display_pictures(self.content_img, self.style_img)

    
    def transfert_style(self, num_epochs, create_gif=False):

        target_style = init_style_target(self.model, self.style_img)
        target_content = init_content_target(self.model, self.content_img)
        self.generated_img = init_generated_img(self.content_img, self.noise_ratio)
            
        start = time()
        for epoch in range(num_epochs) :

            update_style(self.model,
                         target_style, 
                         target_content, 
                         self.generated_img, 
                         self.style_weight, 
                         self.content_weight,
                         self.optimizer)

            if (create_gif == True) :
                add_frame(self.frames, self.generated_img, epoch)
            display_generated_img(self.generated_img, epoch, num_epochs) 
             
        end = time()
        print("Total training time: {:.1f} seconds".format(end-start))
        if (create_gif == True) :
            save_convertion(self.frames, self.content_img, self.style_img, self.generated_img)