import tensorflow as tf

def compute_content_loss(content_generated : dict, content_target : dict, Weight : float, num_style_layers : int):

    loss = 0
    for name in content_generated.keys():

         Generated = content_generated[name]
         Content = content_target[name]
         loss = tf.reduce_mean((Generated - Content) ** 2)

    loss *= (Weight / num_style_layers)

    return (loss)

#%%
def compute_style_loss(style_generated : dict, style_target : dict, Weight : float, num_style_layers : int):

    loss = 0
    for name in style_generated.keys():

        Gram_generated = style_generated[name]
        Gram_style = style_target[name]
        loss = tf.reduce_mean((Gram_generated - Gram_style)**2)

    loss *= (Weight / num_style_layers)
    return (loss)

#%%
def compute_total_loss(content_generated : dict,
               content_target : dict,
               style_generated : dict,
               style_target : dict,
               Weight_content : float,
               Weight_style : float,
               num_style_layers : int):

    content_loss = tf.add_n([compute_content_loss(content_generated,
                                content_target,
                                Weight_content,
                                num_style_layers)])

    style_loss = tf.add_n([compute_style_loss(style_generated,
                            style_target, Weight_style,
                            num_style_layers)])

    total_loss = style_loss + content_loss

    return  (total_loss)
