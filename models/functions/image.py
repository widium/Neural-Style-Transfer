# *************************************************************************** #
#                                                                              #
#    image.py                                                                  #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/10 08:51:01 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** ## =============== Import =================== #
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

from tensorflow import Tensor
from PIL import Image

from cv2 import cvtColor
from cv2 import imread
from cv2 import COLOR_BGR2RGB

from .processing import Normalize_image
from .processing import inverse_normalize_image
from .processing import remove_batch_dimension

# ======================================== #

def get_picture_name(filename : str):

  ch = '.'
  # Remove all characters after the character '-' from string
  name = filename.split(ch, 1)
  name = name[0]
  return (name)
  
# ======================================== #

def load_img_buffer(buffer):
  
  img_pil = Image.open(buffer)
  img_array = np.array(img_pil)
  img = cvtColor(img_array, COLOR_BGR2RGB)
  img = Normalize_image(img)
  
  return (img)

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
  array = remove_batch_dimension(array)
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

