import tensorflow as tf

def flatten_filters(F):
    
    batch = int(F.shape[0])
    flatten_pixel = int(F.shape[1] * F.shape[2])
    filter = int(F.shape[3])
    F = tf.reshape(F, (batch, flatten_pixel, filter))
    return (F)

def normalize_matrix(G, input_tensor):

    height =  tf.cast(input_tensor.shape[1], tf.float32)
    width =  tf.cast(input_tensor.shape[2], tf.float32)
    number_pixels = height * width
    G = G / number_pixels
    return (G)

def gram_matrix(input_tensor):

    F = flatten_filters(input_tensor)
    Gram = tf.matmul(F, F, transpose_a=True)
    Gram = normalize_matrix(Gram, input_tensor)
    return Gram

