# *************************************************************************** #
#                                                                              #
#    save.py                                                                   #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/15 08:32:00 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** #

from tensorflow import Tensor, Variable
from PIL import Image

from .verbose import clear_output, display_convertion_style
from .verbose import display_representation
from .image import tensor_to_image

from ..content_function import display_generated_img

# ======================================== #

def add_frame(frames : list, generated_img, epoch : int):
    
    if (epoch % 5 == 0):
        display_generated_img(generated_img)
        imag = tensor_to_image(generated_img)
        frames.append(imag)

# ======================================== #

def make_gif(frames : list, name : str):
    frame_one = frames[0]
    frame_one.save(f"../img/finish/{name}.gif", 
                   format="GIF", 
                   append_images=frames,
                   save_all=True, 
                   duration=100, 
                   loop=0)

# ======================================== #

def save_convertion(frames : list, content_img : Tensor, style_img : Tensor, generated_img : Variable, name : str):
    
    clear_output(wait=True)
    display_convertion_style(content_img, style_img, generated_img, name)
    make_gif(frames, name)
    
# ======================================== #

def save_evolution(frames : list, img : Tensor, noise_img : Tensor, generated_img : Variable, name : str):
    
    clear_output(wait=True)
    display_representation(img, noise_img, generated_img, name)
    make_gif(frames, name)