# *************************************************************************** #
#                                                                              #
#    streamlit_app.py                                                          #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/12/01 14:12:01 by  ebennace                                 #
#    Updated: 2023/05/04 10:40:43 by Widium                                    #
#                                                                              #
# **************************************************************************** #

import tensorflow as tf
import streamlit as st 

from PIL import Image

from models.neural_style_transfert import NeuralStyleTransfertModel
from models.functions.image import load_img_buffer, load_image
from models.functions.image import tensor_to_image

# **************************************************************************** #

st.set_page_config("Neural Style Transfert", "🎨")
st.header("Neural Style Transfert 🧠 🎨")

print("GPU AVAILABLE ?", tf.config.list_physical_devices('GPU'))
model = NeuralStyleTransfertModel()

# **************************************************************************** #

col1, col2 = st.columns(2)

# content_buffer = col1.file_uploader(label="**Upload Base Image :**")
# style_buffer = col2.file_uploader(label="**Upload Style Image :**")

# if style_buffer != None and content_buffer != None:
    
# model.content_img = load_img_buffer(content_buffer)
# model.style_img = load_img_buffer(style_buffer)

model.content_img = load_image("img/examples/content/content_img.jpg")
model.style_img = load_image("img/examples/style/vangogh_night.jpg")

col1.write(f"Shape : {model.content_img.shape}")
col2.write(f"Shape : {model.style_img.shape}")

col1.image(Image.open("img/examples/content/content_img.jpg"))
col2.image(Image.open("img/examples/style/vangogh_night.jpg"))

model.transfert_style(
    num_epochs=300
)

final = tensor_to_image(model.generated_img)

st.image(final, width=600)
    



# fig, ax = plt.subplots()

# ax.imshow(style_imag)

# st.pyplot(fig)
 