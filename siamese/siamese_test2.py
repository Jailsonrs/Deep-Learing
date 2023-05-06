from tensorflow.keras.layers import Input, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from keras.utils import plot_model

input_shape = (28, 28)
input = Input(shape=input_shape, name="base_input")

def initialize_base_network(input_shape):
    x = Flatten(name="flatten_input")(input)
    x = Dense(128, activation='relu', name="first_base_dense")(x)
    x = Dropout(0.1, name="first_dropout")(x)
    x = Dense(128, activation='relu', name="second_base_dense")(x)
    x = Dropout(0.1, name="second_dropout")(x)
    x = Dense(128, activation='relu', name="third_base_dense")(x)
    return Model(inputs=input, outputs=x)

base_network = initialize_base_network(input_shape)

input_a = Input(shape=input_shape, name="left_input")
vect_output_a = base_network(input_a)

input_b = Input(shape=input_shape, name="right_input")
vect_output_b = base_network(input_b)

output = Dense(1, activation='sigmoid', name="output")(vect_output_a)

siamese_network = Model(inputs=[input_a, input_b], outputs=output)


plot_model(siamese_network, "siamese3.png", True,True, True)