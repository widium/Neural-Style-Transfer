# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    image.py                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/10 08:51:10 by ebennace          #+#    #+#              #
#    Updated: 2022/11/15 15:29:16 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

# =============== Import =================== #
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import Tensor
from PIL import Image

from cv2 import cvtColor
from cv2 import imread
from cv2 import COLOR_BGR2RGB

from function.processing import Normalize_image
from function.processing import inverse_normalize_image
from function.processing import avoid_batch_dimensions

# ======================================== #
  
def load_image(path : str):
    img = imread(path)
    img = cvtColor(img, COLOR_BGR2RGB)
    img = Normalize_image(img)

    return (img)

# ======================================== #

def tensor_to_image(tensor : Tensor):
  tensor = inverse_normalize_image(tensor)
  array = np.array(tensor, dtype=np.uint8)
  array = avoid_batch_dimensions(array)
  img = Image.fromarray(array)
  return img

# ======================================== #

def clip_pixel(image : Tensor):
  cliped_imag = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)
  return (cliped_imag)

# ======================================== #

def create_noisy_imag(img : Tensor, noise_ratio : float):
    
    noise_filter = np.random.randn(*img.shape)
    noise = noise_filter * noise_ratio
    noisy_img = img.copy() + noise
    return (noisy_img)

