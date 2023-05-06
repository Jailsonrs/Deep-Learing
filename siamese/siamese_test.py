import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda, Concatenate, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.utils import plot_model

# Define the Siamese network architecture
INPUT_SHAPE = (28, 28, 1)
LEFT_INPUT  = Input(INPUT_SHAPE, name = 'left_input')
RIGHT_INPUT = Input(INPUT_SHAPE,name = 'right_input')

def base_model():
    input = Input((28,28, ))
    x = Flatten(name = 'flatten_input')(input)
    x = Dense(256, activation = 'relu', name = 'fstbase')(x)
    x = Dropout(0.1, name = 'fstdropout')(x)
    x = Dense(512, activation = 'relu', name = 'secdbase')(x)
    x = Dropout(0.1, name = 'secddropout')(x)
    x = Dense(256, activation = 'relu', name = 'trdbase')(x)
    model = Model(inputs = input, outputs = x)
    return model


base_net = base_model()
encoded_l = base_net(LEFT_INPUT)
encoded_r = base_net(RIGHT_INPUT)

# Use a lambda layer to compute the absolute difference between the encodings
L1_distance = lambda x: tf.keras.backend.abs(x[0] - x[1])
distance = Lambda(L1_distance)([encoded_l, encoded_r])

# Concatenate the encodings and distance and pass through a final dense layer with sigmoid activation
prediction = Dense(1, activation='sigmoid')(Concatenate()([encoded_l, encoded_r, distance]))

model = Model(inputs=[LEFT_INPUT, RIGHT_INPUT], outputs=distance)

# Plot the Siamese network architecture
plot_model(model, to_file='siamese_network.png', show_shapes=True)
