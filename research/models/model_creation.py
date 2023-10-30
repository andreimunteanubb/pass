import keras,math
from keras.applications.inception_v3 import InceptionV3
from keras.layers import GlobalAveragePooling2D,BatchNormalization
from keras.layers.core import Dense,Dropout
from keras.models import Model
import tensorflow as tf
from tensorflow.keras.applications import DenseNet201

def create_model():
    # base_model = InceptionV3(include_top= False, weights="imagenet", input_shape=(224,224,3))
    base_model = tf.keras.applications.efficientnet.EfficientNetB0(include_top=False, weights='imagenet',
                                                                   input_shape=(224, 224, 3))

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    # x = Dense(128, activation= 'relu',kernel_initializer='he_uniform')(x)
    # x = BatchNormalization()(x)
    # x = Dropout(0.3)(x)
    predictions = Dense(4, activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    return model