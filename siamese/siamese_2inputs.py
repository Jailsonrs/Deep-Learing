import tensorflow as tf
from keras.utils import plot_model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.models import Model
from keras.layers import Lambda
import pydantic
from pydantic import BaseConfig as PydanticBaseConfig
from pydantic import BaseModel, validator
import pandas as pd

class BaseArch(tf.keras.layers.Layer):
    INPUT: tf.Tensor = None
    LAYER: tf.Tensor = None
    MODEL: tf.keras.Model
    
    def build(self):
        super(BaseArch, self).__init__()
        self.INPUT = Input(shape = (28,28))
        self.LAYER = Flatten(name = 'flatten_input')(self.INPUT)
        self.LAYER = Dense(256, activation = 'relu', name = 'fstbase')(self.LAYER)
        self.LAYER = Dropout(0.1, name = 'fstdropout')(self.LAYER)
        self.LAYER = Dense(512, activation = 'relu', name = 'secdbase')(self.LAYER)
        self.LAYER = Dropout(0.1, name = 'secddropout')(self.LAYER)
        self.LAYER = Dense(256, activation = 'relu', name = 'trdbase')(self.LAYER)
        self.MODEL = Model(inputs = self.INPUT, outputs = self.LAYER)
        return(self.MODEL)
    



MM = BaseArch()
mm = MM.build()

entradaA = Input((28,28,))
entradaB = Input((28,28,))
mm = mm([entradaA,entradaB])
saida = Dense(1, activation='softmax')(mm)

sNN = Model(inputs = [entradaA, entradaB], outputs = [saida]) 

plot_model(sNN,'snn.png',True, True, True)


def __buld_model__(self): 
    self.corpo = self.NN_BODY
    self.vecA = selfclass Config:
    arbitrary_types_allowed = True.corpo(self.INPUT_A)  
    self.vecB = self.corpo(self.INPUT_B)
    self.Smodel = Model(inputs = [self.INPUT_A, self.INPUT_B]) 
    return(self.Smodel)
    

if __name__ == '__main__':
    BASE_MODEL = init_base_model()
    INPUT_A = Input(shape = (28,28,), name = 'left_in')
    OUTPUT_A = BASE_MODEL(INPUT_A)
    INPUT_B = Input(shape = (28,28,), name = 'right_in' )
    OUTPUT_B = BASE_MODEL(INPUT_B)
    
    DISTANCE = tf.math.reduce_euclidean_norm
    DISTANCE_METRIC = tf.keras.layers.Lambda(DISTANCE)([OUTPUT_A, OUTPUT_B])
    FINAL_MODEL = Model(inputs = [INPUT_A, INPUT_B], outputs = [DISTANCE_METRIC])
    plot_model(FINAL_MODEL, "finalmodel.png",True, True, True)
