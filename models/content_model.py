# *************************************************************************** #
#                                                                              #
#    content_model.py                                                          #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/15 13:15:03 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from tensorflow.keras.optimizers import Adam

from tqdm import tqdm

import matplotlib.pyplot as plt 

from time import time

from .functions.image import get_picture_name
from .functions.image import load_image

from .functions.save import add_frame
from .functions.save import save_evolution

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
    def __init__(self, optimizer=Adam(learning_rate=0.02)):
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
    
    def recreate_content(self, num_epochs, create_gif=False, name : str = "content_representation"):

        target_content = init_content_target(self.model, self.content_img)
        self.generated_img = init_generated_img(self.content_img)
        
        start = time()
        for epoch in tqdm(range(num_epochs)) :

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
            save_evolution(self.frames, self.content_img, self.noise_img, self.generated_img, name)
            print(f"Gif saved in finish/{name}.gif")
            print(f"Subplot saved in finish/{name}.png")