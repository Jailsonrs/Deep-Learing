 
from keras import Sequential
from keras.layers import Dense
import numpy as np
from keras import losses

xs = np.array([-1,0, 0.0, 1.0, 2.0, 3.0, 4.0])
ys = np.array([-3,0, -1.0, 1.0, 3.0, 5.0, 7.0])

model = Sequential([Dense(units=1,input_shape = [1])])

loss_fn = losses.MeanSquaredError()
model.compile(optimizer = 'sgd', loss = loss_fn)

model.fit(xs, ys, epochs = 1000)
 
print(model.predict([10.0]))