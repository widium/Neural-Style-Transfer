import tensorflow as tf
import matplotlib.pyplot as plt
from keras.applications.vgg19 import preprocess_input
import numpy as np
import PIL.Image

# def load_image(image):
#     image = plt.imread(image)
#     img = tf.image.convert_image_dtype(image, tf.float32)
#     img = tf.image.resize(img, [400, 400])
#     # Shape -> (batch_size, h, w, d)
#     img = img[tf.newaxis, :]
#     return img

def load_image(path_to_img):

    max_dim = 512
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_image(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)

    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

# def load_image(img_path):
#     img = tf.io.read_file(img_path)
#     img = tf.io.decode_jpeg(img, channels = 3)
#     img = tf.image.resize(img, size = (224, 224), method = 'bicubic')
#     img = img / 255.0
#     return img

def processing_image(image):

    image = image * 255.0
    image = preprocess_input(image)
    return (image)

def clip_pixel(image):

    cliped_image = tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=1.0)

    return (cliped_image)

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)