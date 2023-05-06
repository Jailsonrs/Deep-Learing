import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from keras.models import Model
from keras.layers import Dense, Input
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
import argparse


parser = argparse.ArgumentParser(prog = 'funcNN')

parser.add_argument('-e', '--epochs' ,type = int, help = 'set the number of epochs for training the model')
args = parser.parse_args()



##utility functions
def format_output(data):
    y1 = data.pop('Y1')
    y1 = np.array(y1)
    y2 = data.pop('Y2')
    y2 = np.array(y2)
    return y1, y2


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


def plot_diff(y_true, y_pred, title=''):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.xlim(plt.xlim())
    plt.ylim(plt.ylim())
    plt.plot([-100, 100], [-100, 100])
    plt.show()


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0, ylim)
    plt.plot(history.history[metric_name], color='blue', label=metric_name)
    plt.plot(history.history['val_' + metric_name], color='green', label='val_' + metric_name)
    plt.show()

#loading and preparing the dataset
# Specify data URI
URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx'

# Use pandas excel reader
df = pd.read_excel(URL)
df = df.sample(frac=1).reset_index(drop=True)

# Split the data into train and test with 80 train / 20 test
train, test = train_test_split(df, test_size=0.2)
train_stats = train.describe()
# Get Y1 and Y2 as the 2 outputs and format them as np arrays
train_stats.pop('Y1')
train_stats.pop('Y2')
train_stats = train_stats.transpose()
train_Y = format_output(train)
test_Y = format_output(test)

# Normalize the training and test data
norm_train_X = norm(train)
norm_test_X = norm(test)

#definindo a NN
entrada = Input(shape = (len(train.columns),))

h1 = Dense(256, activation='relu')(entrada)
h2 = Dense(256, activation='relu')(h1)
y1 = Dense(1, name = 'y1out')(h2)
h3 = Dense(128, activation = 'relu')(h2)
y2 = Dense(1, name = 'y2out')(h3)

final_model = Model(inputs = entrada, outputs = [y1, y2])
plot_model(final_model, show_layer_names = True, show_shapes = True,    to_file='func.png')

optimizer = tf.keras.optimizers.SGD(learning_rate = 0.001)
final_model.compile(optimizer=optimizer,
              loss={'y1out': 'mse', 'y2out': 'mse'},
              metrics={'y1out': tf.keras.metrics.RootMeanSquaredError(),
                       'y2out': tf.keras.metrics.RootMeanSquaredError()})
history = final_model.fit(norm_train_X, train_Y,
                batch_size = 30,
                epochs = args.epochs,
                verbose = 1,
                validation_data = (norm_test_X, test_Y))


# Test the model and print loss and mse for both outputs
loss, Y1_loss, Y2_loss, Y1_rmse, Y2_rmse = final_model.evaluate(x= norm_test_X, y=test_Y)
print("Loss = {}, Y1_loss = {}, Y1_mse = {}, Y2_loss = {}, Y2_mse = {}".format(loss, Y1_loss, Y1_rmse, Y2_loss, Y2_rmse))

# Plot the loss and mse
Y_pred = final_model.predict(norm_test_X)
plot_diff(test_Y[0], Y_pred[0], title='Y1')
plot_diff(test_Y[1], Y_pred[1], title='Y2')
plot_metrics(metric_name='y1out_root_mean_squared_error', title='Y1 RMSE', ylim=6)
plot_metrics(metric_name='y2out_root_mean_squared_error', title='Y2 RMSE', ylim=7)