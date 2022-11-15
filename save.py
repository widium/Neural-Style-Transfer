# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    save.py                                            :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/11/15 08:32:06 by ebennace          #+#    #+#              #
#    Updated: 2022/11/15 08:32:45 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

from PIL import Image

from verbose import clear_output, display_convertion_style
from image import tensor_to_image

# ======================================== #

def add_frame(frames : list, generated_img):
    imag = tensor_to_image(generated_img)
    frames.append(imag)

# ======================================== #

def make_gif(frames : list):
    frame_one = frames[0]
    frame_one.save("img/finish/evolution.gif", 
                   format="GIF", 
                   append_images=frames,
                   save_all=True, 
                   duration=100, 
                   loop=0)

# ======================================== #

def save_convertion(frames, content_img, style_img, generated_img):
    
    clear_output(wait=True)
    display_convertion_style(content_img, style_img, generated_img)
    make_gif(frames)
    
# ======================================== #