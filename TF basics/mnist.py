import tensorflow as tf
from keras import losses
from keras import Sequential
from keras import layers

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
x_train = x_train/255

model = Sequential([
    layers.Conv2D(12, (3,3),activation = 'relu', input_shape = (28, 28, 1)),
    layers.AveragePooling2D(4,4),
    layers.Flatten(),
    layers.Dense(units = 128 , activation = tf.nn.relu),
    layers.Dense(units = 10, activation = tf.nn.softmax)
    ])
    
model.compile(optimizer = "sgd", 
              loss=tf.losses.SparseCategoricalCrossentropy(), 
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs = 9)



