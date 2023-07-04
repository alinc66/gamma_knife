import streamlit as st
import cv2
import numpy as np
import base64
import os
from PIL import Image
from keras.applications.vgg19 import preprocess_input
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras.layers import Dense, Dropout, Flatten
from keras.models import Model

import subprocess
if not os.path.isfile('model.hdf5'):
    subprocess.run(['curl --output model.hdf5 "https://media.githubusercontent.com/media/alincbuz/gamma_knife/main/tune_model19.weights.best_2.hdf5"'], shell=True)
    
st.markdown('<h1 style="color:black;">AI imaging prognostic factors in the evolution of stage-treated metastases using Gamma Knife</h1>', unsafe_allow_html=True)
st.markdown('<h2 style="color:gray;">The image classification model classifies image into following categories:</h2>', unsafe_allow_html=True)
st.markdown('<h3 style="color:gray;"> tumor regression - class 1, tumor progression - class 0 </h3>', unsafe_allow_html=True)


# background image to streamlit

@st.experimental_memo()
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: scroll; # doesn't work
    }
    </style>
    ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)
    return

set_png_as_page_bg('background.webp')

upload = st.file_uploader('Insert image for classification', type=['png', 'jpg'])
c1, c2 = st.columns(2)
if upload is not None:
    im = Image.open(upload)
    img = np.asarray(im)
    image = cv2.resize(img, (224, 224))
    img = preprocess_input(image)
    img = np.expand_dims(img, 0)
    c1.header('Input Image')
    c1.image(im)
    #c1.write(img.shape)


    def model(input_shape, n_classes, optimizer='adam', fine_tune=0):
        """
        Complies a model integrated with vgg16 pretrained layers

        Parameters
        ------------------------
        input_shape: tuple
        shape of input images (width, height, channels)
        n_classes: int
        Number of classes for output layer
        optimizer: string
        Instantiated optimizer for training, default to adam
        fine_tune: int
        No of pretrained layers to unfreeze.
        If set to 0, all layers will freeze during training

        Return
        -------------------------------
        model: compiled model

        """

        # pretrained layers are added. Include_top is set to false, in order to exclude model's fully connected layers.
        conv_base = VGG19(weights='imagenet',
                          include_top=False,
                          input_shape=input_shape)

        # Defines how many layers to freeze.
        # Layers in conv_base are changed from trainable to non-trainable based on fine_tune value.
        if fine_tune > 0:
            for layer in conv_base.layers[:-fine_tune]:
                layer.trainable = False

        else:
            for layer in conv_base.layers:
                layer.trainable = False

        # create fully-connected layers
        top_model = conv_base.output
        top_model = Flatten(name='flatten')(top_model)
        top_model = Dense(4096, activation='relu')(top_model)
        top_model = Dense(1072, activation='relu')(top_model)
        top_model = Dropout(0.2)(top_model)
        output_layer = Dense(n_classes, activation='softmax')(top_model)

        # group pretrained layers and fully connected layers.
        model = Model(inputs=conv_base.input, outputs=output_layer)

        # complies model
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model


    # load weights of the trained model.
    input_shape = (224, 224, 3)
    optim_1 = Adam(learning_rate=0.0001)
    n_classes = 2
    vgg_model = model(input_shape, n_classes, optim_1, fine_tune=2)
    vgg_model.load_weights('model.hdf5')

    # prediction on model
    vgg_preds = vgg_model.predict(img)
    vgg_pred_classes = np.argmax(vgg_preds, axis=1)
    c2.header('Output')
    c2.subheader('Predicted class :')
    #c2.write(vgg_pred_classes[0])
    c2.write(f'{vgg_pred_classes[0]}')
