import tensorflow as tf

def flatten_features(F):
    content = int(F.shape[-1])
    F = tf.reshape(F, [-1, content])
    return (F)

def normalize_matrix(G, F):

    number = tf.shape(F)[0]
    number = tf.cast(number, tf.float32)
    G = G / number
    return (G)

def gram_matrix(F):

    F = flatten_features(F)
    Gram = tf.matmul(F, F, transpose_a=True)
    Gram = normalize_matrix(Gram, F)
    return Gram

