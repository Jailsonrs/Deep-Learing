import tensorflow as tf
from keras.layers import Input
from keras.models import Model
from keras.utils import plot_model

x = tf.keras.Input(shape = (300,28))
x_0 = tf.keras.layers.Dense(128)(x)
x_1 = tf.keras.layers.Dense(256, activation = 'relu')(x_0)
x_2 = tf.keras.layers.Dense(512, activation = 'relu')(x_0)

merge = tf.keras.layers.Concatenate()([x_1, x_2])
X = tf.keras.layers.Flatten()(merge)

y_1 = tf.keras.layers.Dense(units = 1,   activation = 'sigmoid')(X)
y_2 = tf.keras.layers.Dense(units = 1,   activation = 'sigmoid')(X)

func_model = Model(inputs = x, outputs = [y_1, y_2])

plot_model(func_model, show_shapes=True, show_layer_names=True, to_file='model.png')
