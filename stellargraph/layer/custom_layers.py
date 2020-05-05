from tensorflow.keras import backend as K
import tensorflow as tf

class Normalization(tf.keras.layers.Layer): 

    def __init__(self): 
        super(Normalization, self).__init__()
        
    def call(self, inputs): 
        return inputs



class Squeeze(tf.keras.layers.Layer):

    def __init__(self, inputs): 
        super(Squeeze, self).__init__()
        self.axis = 0 
        self.inputs = inputs


    def call(self, x): 
        return K.squeeze(self.inputs, self.axis)(x)
