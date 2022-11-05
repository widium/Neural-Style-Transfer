import tensorflow as tf
from image import processing_image
import numpy as np

# def gram_matrix(input_tensor):
#     channels = int(input_tensor.shape[-1])
#     a = tf.reshape(input_tensor, [-1, channels])
#     n = tf.shape(a)[0]
#     gram = tf.matmul(a, a, transpose_a=True)
#     return gram / tf.cast(n, tf.float32)

def gram_matrix(input_tensor):
  result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
  input_shape = tf.shape(input_tensor)
  num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
  return result/(num_locations)

def features_maps_extractor(layer_names):

    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False

    outputs = [vgg.get_layer(name).output for name in layer_names]

    model = tf.keras.Model([vgg.input], outputs)

    return model

class StyleContentModel(tf.keras.models.Model):

    def __init__(self, style_layers, content_layers):
        super().__init__()
        self.vgg = features_maps_extractor(style_layers + content_layers)
        self.style_layers = style_layers
        self.content_layers = content_layers
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs):
        "Expects float input in [0,1]"
        inputs = inputs * 255.0
        preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)

        outputs = self.vgg(preprocessed_input)

        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        content_dict = {content_name: value for content_name, value in
                        zip(self.content_layers, content_outputs)}

        style_dict = {style_name: value for style_name, value in
                      zip(self.style_layers, style_outputs)}

        return {'content': content_dict, 'style': style_dict}



class Content_Extractor(tf.keras.models.Model):

  def __init__(self, content_layers):

    super().__init__()
    init_layer = ['block1_conv1']
    self.vgg = features_maps_extractor(init_layer + content_layers)
    self.vgg.trainable = False
    self.content_layers = content_layers
    self.num_style_layers = len(init_layer)

  def call(self, inputs):

    inputs = inputs * 255.0
    preprocessed_input = tf.keras.applications.vgg19.preprocess_input(inputs)
    outputs = self.vgg(preprocessed_input)
    content_outputs = outputs[self.num_style_layers:]

    content_dict = {content_name: value
                    for content_name, value
                    in zip(self.content_layers, content_outputs)}

    return content_dict

def update_shape_in_dict(content_dict):

    for name, output in content_dict.items():
        nbr_shape = len(output.shape.as_list())
        if (nbr_shape == 3):
            output = output[tf.newaxis, :]
            content_dict[name] = output
    return content_dict

class Style_Extractor(tf.keras.models.Model):

    def __init__(self, style_layers):
        super().__init__()
        self.extractor = features_maps_extractor(style_layers)
        self.style_layers = style_layers
        self.num_style_layers = len(style_layers)
        self.extractor.trainable = False


    def call(self, inputs):

        preprocessed_input = processing_image(inputs)

        style_outputs = self.extractor(preprocessed_input)

        style_outputs = [gram_matrix(style) for style in style_outputs]

        style_dict = {style_name: value for style_name, value in zip(self.style_layers, style_outputs)}

        return style_dict
