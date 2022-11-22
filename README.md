# Neural-Style-Transfer
## Example and Notebook
- [Example of Convertion](/img/result/)
- [Notebook](/notebook/README.md)
    - [Neural Style Transfert Notebook](style_transfert.ipynb)
    - [Recreate Content Notebook](recreate_content.ipynb)
    - [Recreate Style Notebook](recreate_style.ipynb)
***
## Understanding Project 
- [1. Features Map](#features-maps)
    - [1. Import VGG19](#import-vgg19-model)
    - [2. Create List of Convolution Layers name](#create-list-of-convolution-layers-name)
    - [3. Create Model who output list of features map](#create-model-who-output-list-of-features-map)
    - [4 . Import and Preprocess Imag](#import-and-preprocess-image)
    - [5. Plot one filter for each features Maps](#plot-one-filter-for-each-features-maps)
- [2. Cost Function](#cost-function)
    - [1. Content Cost Function](#content-cost-function)
        - [1. Create Custom Model to Generate One Features Map](#create-custom-model-to-generate-one-features-map)
        - [2. Compute Error With Features Maps](#compute-error-with-features-maps)
        - [3. Recreate Content with Features Maps](#recreate-content-with-features-maps)
    - [2. Style Cost Function](#style-cost-function)
        - [1. Create Custom Model thats output "list of Features Maps"](#create-custom-model-thats-output-list-of-features-maps)
        - [2. Extract Style](#extract-style)
            - [(Optional) Gram Matrix](#gram-matrix)
            - [1. Filter Map to Matrix of Pixel](#filter-map-to-matrix-of-pixel)
            - [2. Create Gram Style Matrix](#create-gram-style-matrix)
            - [3. Get Entire Style Of Image]()
        - [3. Compute Error Between 2 List of Gram Matrix](#compute-error-between-2-list-of-gram-matrix)
        - [4. Recreate Style](#recreate-style)
    - [3. Total Cost Function](#total-cost-function)
        - [1. Recreate Content with Style](#recreate-content-with-style)
***
<p align="center">
    <img src="https://i.imgur.com/F2eZCTV.gif">
    <img src="https://i.imgur.com/CX1oilh.png">
<p>

***
# Features Maps
- [1. Import VGG19](#import-vgg19-model)
- [2. Create List of Convolution Layers name](#create-list-of-convolution-layers-name)
- [3. Create Model who output list of features map](#create-model-who-output-list-of-features-map)
- [4 . Import and Preprocess Imag](#import-and-preprocess-image)
- [5. Plot one filter for each features Maps](#plot-one-filter-for-each-features-maps)

![](https://i.imgur.com/oqaOEed.png)

- ## Visualize feature extraction in a CNN

- ### Import VGG19 Model
~~~python
def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

vgg19 = load_vgg19()
vgg19.summary()

>>
=================================================================
Total params: 143,667,240
Trainable params: 143,667,240
Non-trainable params: 0
_________________________________________________________________
~~~
- ### Create List of Convolution Layers name
~~~python
def create_list_of_vgg_layer():

    layers_name   = ['block1_conv1',
                     'block2_conv1',
                     'block3_conv1',
                     'block4_conv1',
                     'block5_conv1']

    return (layers_name)
~~~
### Create Model who output list of features map
- **iterate** in list of layers name 
- **get** output shape of each layer in vgg19
- **append** in list of outputs
- **define** the New Model with a list of features map as output
~~~python
def create_multi_output_model(layer_name : list)-> Model:

    vgg19 = load_vgg19()
    list_of_features_map = list()
    
    for name in layers_name:
        layer = vgg19.get_layer(name)
        output = layer.output
        list_of_features_map.append(output)

    model = Model([vgg19.input], list_of_features_map)
    model.trainable = False

    return (model)
~~~
### Import and Preprocess image
- `Load image` - Keras
- **Preprocess** array with with the specialize function of the VGG19 model
- **Recover** the list of features maps 
~~~python
img = load_img('waves.jpg')  
img = img_to_array(img)  
img = expand_dims(img, axis=0)  
input = preprocess_input(img)  
feature_maps = model.predict(input)
~~~

### Plot one filter for each features Maps
- **Define** the size of `Subplot`
- **Iterate** in list of features mpas
- **Plot** one filter in features map `Tensor` with `Imshow`
~~~python
fig, ax = plt.subplots(1, 5, figsize=(20, 15))

i = 0
for f in feature_maps :
    ax[i].imshow(f[0, :, :, 4], cmap='gray')
    i += 1
~~~
![](https://i.imgur.com/4Z0nRjH.png)

# Cost Function
- [1. Content Cost Function](#content-cost-function)
- [2. Style Cost Function](#style-cost-function)
- [3. Total Cost Function](#total-cost-function)

# Content Cost Function
- ### [Recreate Content Model Notebook](/notebook/recreate_content.ipynb)
***
<p align="center">
    <img src="https://i.imgur.com/TAuDx1e.gif">
<p>

- [1. Create Custom Model to Generate One Features Map](#create-custom-model-to-generate-one-features-map)
- [2. Compute Error With Features Maps](#compute-error-with-features-maps)
- [3. Recreate Content with Features Maps](#recreate-content-with-features-maps)


## Learn to Recreate Content
- To recreate an image we will base it on the production of Features Map.
- We pass 2 image in the model :
	- *One with random pixels  $\large G$*
	- *One with a content $\large C$*
- We get the output of one convolution of the model and we compare **The value of the Pixel in all Filter for the 2 images by** :
	- calculating **the difference between the 2 Tensor of Features Map $F$ and $P$**

$$\Large L_\text {content}(G, C)=\frac{1}{2} \sum(G - C)^{2}$$

![](https://i.imgur.com/nGlKPq6.jpg)
## Create Custom Model to Generate One Features Map
![](https://i.imgur.com/6dLki5Y.png)
- From the VGG19 Network we recreate a Model that outputs 1 feature maps of a given image
- **Load** the VGG Network without the Fully Connected Layer (FC) and without the Output Layer
- **Define** the output of the New Model as a Convolution Layer
- **Set** the Un-trainaible parameters
~~~python
content_layers = ['block2_conv2']

def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

def create_model(content_layers : list)-> Model:

    vgg19 = load_vgg19()
    name = content_layers[0]
    layer = vgg19.get_layer(name)
    output = layer.output

    model = Model([vgg19.input], output)
    model.trainable = False

    return (model)

model = create_model(content_layers)
~~~
~~~python
model.summary()

_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_5 (InputLayer)        [(None, None, None, 3)]   0         
                                                                 
 block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                 
 block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                 
 block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                 
 block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                 
=================================================================
Total params: 260,160
Trainable params: 0
Non-trainable params: 260,160
_________________________________________________________________
~~~
## Get Features Maps Custom Model
- **Preprocessing** img for Custom Model 
- **Get** Output of Custom Model `features Maps`
~~~C
def get_features_map(model : Model, img : Tensor)->list:

	process_img = preprocessing_img(img)
	features_map = model(process_img)

    return (features_map)
~~~

## Compute Error With Features Maps
- For the Model can compare 2 images we give it feature maps
- We can calculate **the pixel difference between the 2 feature maps** with a Square Error MSE

~~~python
 def compute_content_loss(content_generated : Tensor, content_target : Tensor):
    
    content_loss = tf.reduce_mean((content_generated - content_target)**2)
    return (content_loss)
~~~

![](https://i.imgur.com/uCpB9Ih.png)

## Recreate Content with Features Maps
**For each Iteration :**
- **Extract** Content Features Maps
- **Compute** Error with `Target Content`
- **Update** `Generated Image`

![](https://i.imgur.com/bXYAOaK.jpg)


# Style Cost Function 
- ### [Recreate Style Model Notebook](/notebook/recreate_style.ipynb)


<p align="center">
  <img src="gif/style_evolution.gif">
</p>

- [1. Create Custom Model thats output "list of Features Maps"](#create-custom-model-thats-output-list-of-features-maps)
- [2. Extract Style](#extract-style)
- [3. Compute Error Between 2 List of Gram Matrix](#compute-error-between-2-list-of-gram-matrix)
- [4. Recreate Style](#recreate-style)

## *Learn to Recreate Style* 
- To recreate Style we need to have **multiple features map for one image** 
- **Compute the Correlation Between Filter** of all Features maps for understand the paterns in style (*List of Gram Matrix*)
- **Initialise** Image with Random PIxel `Generated Image`
- **Set** the `Target Style` with the *List of Gram Matrix* of `Style Image`
- **Compute** the Difference of **2** *list of Gram Matrix* (`Style Image` and` Generated Image`)
- **For** each update of `Generated Image` :
	- **get** the *List of Gram Matrix* of `Generated Image`
	- **Compare** with `Style Target `
	- **Update** `Generated Image`

### $$G = \text{Gram Matrix of Generated Image}$$
### $$S = \text{Gram Matrix of Style Image}$$
## $$L_{\text{Style}}(G, S)=\frac{1}{2} \sum(G - S)^{2}$$
![](https://i.imgur.com/0GEgvGm.jpg)

***
## Create Custom Model thats output "list of Features Maps"
![](https://i.imgur.com/gg7tHqC.png)

### Create List of Convolution Layer Output
- List all Convolution Layer that we want to return 

~~~python
def create_list_of_vgg_layer():

    style_layer_names   = ['block1_conv1',
                           'block2_conv1',
                           'block3_conv1',
                           'block4_conv1',
                           'block5_conv1']

    return (style_layer_names)
~~~

##### Load VGG19
- **Load** the VGG Network without the Fully Connected Layer and without Output Layer

~~~python
def load_vgg19()-> Model:
    vgg = VGG19(include_top=False, weights='imagenet')
    return vgg

~~~

### Create Custom Model 

-   **iterate** in list of layers name
-   **get** output shape of each layer in vgg19
-   **append** in list of outputs
-   **define** the New Model with a list of features map as output

```python
def create_multi_output_model(layer_name : list)-> Model:

    vgg19 = load_vgg19()
    list_of_features_map = list()
    
    for name in layers_name:
        layer = vgg19.get_layer(name)
        output = layer.output
        list_of_features_map.append(output)

    model = Model([vgg19.input], list_of_features_map)
    model.trainable = False

    return (model)
```

~~~python
model = create_multi_output_model(layers_name)
model.summary()
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 input_2 (InputLayer)        [(None, None, None, 3)]   0         
                                                                 
 block1_conv1 (Conv2D)       (None, None, None, 64)    1792      
                                                                 
 block1_conv2 (Conv2D)       (None, None, None, 64)    36928     
                                                                 
 block1_pool (MaxPooling2D)  (None, None, None, 64)    0         
                                                                 
 block2_conv1 (Conv2D)       (None, None, None, 128)   73856     
                                                                 
 block2_conv2 (Conv2D)       (None, None, None, 128)   147584    
                                                                 
 block2_pool (MaxPooling2D)  (None, None, None, 128)   0         
                                                                 
 block3_conv1 (Conv2D)       (None, None, None, 256)   295168    
                                                                 
 block3_conv2 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_conv3 (Conv2D)       (None, None, None, 256)   590080    
                                                                 
 block3_conv4 (Conv2D)       (None, None, None, 256)   590080    
...
Total params: 12,944,960
Trainable params: 0
Non-trainable params: 12,944,960
_________________________________________________________________
~~~

![](https://i.imgur.com/0GEgvGm.jpg)


## Extract Style 
- [(Optional) Gram Matrix](#gram-matrix)
- [1. Filter Map to Matrix of Pixel](#filter-map-to-matrix-of-pixel)
- [2. Create Gram Style Matrix](#create-gram-style-matrix)
- [3. Get Entire Style Of Image]()

***
## Gram Matrix
### Matrix of Correlation between Vectors
- ### Dot Product with $\large n$ Vector

```math
G(\{v_{1},\dots ,v_{n}\})={\begin{vmatrix}\langle v_{1},v_{1}\rangle &\langle v_{1},v_{2}\rangle &\dots &\langle v_{1},v_{n}\rangle \\\langle v_{2},v_{1}\rangle &\langle v_{2},v_{2}\rangle &\dots &\langle v_{2},v_{n}\rangle \\\vdots &\vdots &\ddots &\vdots \\\langle v_{n},v_{1}\rangle &\langle v_{n},v_{2}\rangle &\dots &\langle v_{n},v_{n}\rangle \end{vmatrix}}
```

![](https://i.imgur.com/IGkOclV.png)

- ### Correlation between One matrix and its tranpose
~~~python
def gram_matrix(F):

    Gram = tf.matmul(F, F, transpose_a=True)
    return Gram
~~~
***
## Filter Map to Matrix of Pixel
- **Flatten** each **Filter** in Features maps to **Vector of Pixel** 
- **Create** Matrix with `N` **Vector of Pixel** 

~~~python
def flatten_filters(Filters):
    
    batch = int(Filters.shape[0])
    vector_pixels = int(Filters.shape[1] * Filters.shape[2])
    nbr_filter = int(Filters.shape[3])
    
    matrix_pixels = tf.reshape(Filters, (batch, vector_pixels, nbr_filter))
    return (matrix_pixels)

~~~

#### $$\Large F = \text{Matrix of Vector Pixels}$$
![](https://i.imgur.com/C02PB9F.png)

## Create Gram Style Matrix 

- **Compute** the Correlation between each Filter 
- **Get** the Transpose of $\large F$ $(\large F^{T})$
- **Make** **the Dot Product** between each Vector in $\large F \text{ with } F^{T}$
- **Normalize** Value with the number of pixel 

~~~python
def gram_matrix(Filters):

    F = flatten_filters(Filters)
    Gram = tf.matmul(F, F, transpose_a=True)
    Gram = normalize_matrix(Gram, Filters)
    return Gram

def normalize_matrix(G, Filters):

    height =  tf.cast(Filters.shape[1], tf.float32)
    width =  tf.cast(Filters.shape[2], tf.float32)
    number_pixels = height * width
    G = G / number_pixels
    return (G)
~~~

![](https://i.imgur.com/CqWxvoF.png)

## Get Entire Style Of Image

- **Get** List of Features maps 
- **Convert** each Filters to Vector of Pixel
- **Compute** Gram Matrix for Each Feature Map
- **Save** List of Gram Matrix 
- **`The list of Gram matrices will become our target`**

~~~python
def extract_style(features_map):

    Grams_styles = list()
    
    for style in features_map:
        Gram = gram_matrix(style)
        Grams_styles.append(Gram)
    return Grams_styles
~~~
![](https://i.imgur.com/32vSl4E.png)

## Compute Error Between 2 List of Gram Matrix
- **Get** List of Features Maps for Style Image and Generated Image
- **Compute** Gram Style Matrix for Each Features Maps for the 2 Image
- **Calculate** the difference between the 2 Gram Matrix lists

~~~python
def compute_style_loss(style_generated : Tensor, 
                       style_target : Tensor):

    all_style_loss = list()

    for generated, target in zip(style_generated, style_target):

        style_layer_loss = tf.reduce_mean((generated - target)**2)
        all_style_loss.append(style_layer_loss)

    num_style_layers = len(all_style_loss)
    style_loss = tf.add_n(all_style_loss) / num_style_layers

    return (style_loss)
~~~
![](https://i.imgur.com/KNM2KPU.png)

## Recreate Style 
**For** each iteration
- **Get** List of Features Maps of `Generated Image`
- **Compute** The Gram Style Matrix for each Features Maps
- **Compute** Loss With :
	- *Gram Style Matrix of Generated Image*
	- *Gram Style Matrix of Style Image (Target)*
- **Update** Pixel of Image

![](https://i.imgur.com/442pRIp.jpg)


## Total Cost Function
- ### [Style Transfert Model Notebook](/notebook/style_transfert.ipynb)

<p align="center">
    <img src="https://i.imgur.com/F2eZCTV.gif">
<p>

- [1. Recreate Content with Style](#recreate-content-with-style)
***
- **Extract** Content and Style for Generated Image and the Target Image 
- **Compute** Total Loss With the Addition between Style Loss and Content Loss
	- [Style Cost Function](#style-cost-function)
	- [Content Cost Function](#content-cost-function)
- **Weighting** each Loss to prioritize the style or the content
### $$\LARGE L_{\text {Total}}=\theta . L_{\text {Content}}+\beta . L_{\text {Style}}$$

![](https://i.imgur.com/21OPtvs.png)

## Recreate Content with Style
### Compute Total loss for the generated image for each iteration
- **Extract** Style in `Generated Image`
- **Extract** Content `Generated Image`
- **Compute** Total loss With `Target Style` $\Large S$ and `Target Content` $\Large C$
- **Minimize** the Error 

![](https://i.imgur.com/w6fV6Eo.jpg)