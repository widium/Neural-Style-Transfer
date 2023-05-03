# *************************************************************************** #
#                                                                              #
#    style.py                                                                  #
#                                                                              #
#    By: Widium <ebennace@student.42lausanne.ch>                               #
#    Github : https://github.com/widium                                        #
#                                                                              #
#    Created: 2022/11/10 09:53:03 by  ebennace                                 #
#    Updated: 2023/05/03 16:05:48 by Widium                                    #
#                                                                              #
# **************************************************************************** ## =============== Import =================== #
import tensorflow as tf
from datetime import datetime

from tensorflow import Tensor
from tensorflow import GradientTape 
from keras import Model

from tensorflow import Variable
from tensorflow.keras.optimizers import Optimizer

from .vgg import get_features_map
from .loss import compute_total_loss
from .compute import compute_and_optimize_gradient
from .extract import extract_content, extract_style

# ======================================== #
@tf.function
def update_style(model : Model,
                 style_target : Tensor, 
                 content_target : Tensor, 
                 generated_img : Variable,
                 style_weight : float,
                 content_weight : float, 
                 optimizer : Optimizer):

    with GradientTape() as tape :

        features_map = get_features_map(model, generated_img)
        style_generated = extract_style(features_map)
        content_generated = extract_content(features_map)
        
        loss = compute_total_loss(style_generated,
                                  content_generated,
                                  style_target,
                                  content_target,
                                  style_weight,
                                  content_weight)

    generated_img = compute_and_optimize_gradient(tape,
                                                  optimizer, 
                                                  generated_img,
                                                  loss)
    
# ======================================== #