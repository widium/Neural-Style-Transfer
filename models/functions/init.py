# *************************************************************************** #
#                                                                              #
#    init.py                                                                   #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/10 09:18:03 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** ## =============== Import =================== #
import tensorflow as tf
import numpy as np


from tensorflow import Tensor

from tensorflow import Variable
from keras import Model

from .vgg import get_features_map
from .processing import create_batch_image
from .image import create_noisy_imag, clip_pixel

from .extract import extract_content, extract_style

# ======================================== #

def init_content_target(model : Model, content_img : Tensor):

        content_img = create_batch_image(content_img)
        features_map = get_features_map(model, content_img)
        content_target = extract_content(features_map)

        return (content_target)

# ======================================== #

def init_style_target(model : Model, style_img : Tensor):

    style_img = create_batch_image(style_img)
    features_map = get_features_map(model, style_img)
    style_target = extract_style(features_map)

    return (style_target)



# ======================================== #

def init_generated_img(content_img : Tensor, 
                       noise_ratio : float = 0.35):

    
    generated_img = create_noisy_imag(content_img, noise_ratio)
    generated_img = clip_pixel(generated_img)
    generated_img = create_batch_image(generated_img)
    generated_img = Variable(generated_img)

    return (generated_img)