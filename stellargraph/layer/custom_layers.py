from tensorflow.keras.backend import K 
import tensorflow as tf

class Normalization(tf.keras.layers.Layer): 

    def __init__(self): 
        super(Normalization, self).__init__()
        
    def call(self, inputs): 
        return inputs



class Squeeze(tf.keras.layers.Layer):

    def __init__(self): 
        super(Dense, self).__init__()
        self.axis = 0

    def call(self, inputs): 
        return K.squeeze(inputs, self.axis)