# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    app.py                                             :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: ebennace <ebennace@student.42lausanne.c    +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2022/12/01 14:12:13 by ebennace          #+#    #+#              #
#    Updated: 2022/12/06 18:32:11 by ebennace         ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

import sys
sys.path.append("..")

import pandas as pd 
import numpy as np

from PIL import Image

import streamlit as st 
import matplotlib.pyplot as plt

from time import sleep

from backend.model.neural_style_transfert import Model_Style_Transfert
from backend.function.image import load_image
from backend.function.image import tensor_to_image

st.set_page_config("Neural Style Transfert", "🎨")

st.header("Neural Style Transfert 🧠 🎨")

Model = Model_Style_Transfert()

col1, col2 = st.columns(2)

content_buffer = col1.file_uploader("**Upload Base Image :**")
style_buffer = col2.file_uploader("**Upload Style Image :**")

if style_buffer != None and content_buffer != None:
    Model.import_img(content_buffer, style_buffer)
    col1.write(f"Shape : {Model.content_img.shape}")
    col2.write(f"Shape : {Model.style_img.shape}")
    col1.image(content_buffer)
    col2.image(style_buffer)
    Model.transfert_style(num_epochs=5)
    final = tensor_to_image(Model.generated_img)
    st.image(final, width=600)
    



# fig, ax = plt.subplots()

# ax.imshow(style_imag)

# st.pyplot(fig)
 