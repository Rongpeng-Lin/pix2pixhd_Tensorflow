from scipy import misc
import numpy as np
import tensorflow as tf
import math
import functools
import os

def ins_norm(name,x):
    with tf.variable_scope(name):
        return tf.contrib.layers.instance_norm(x)
    
def relu(name,x):
    with tf.variable_scope(name):
        return tf.nn.relu(x)
    
def tanh(name,x):
    with tf.variable_scope(name):
        return tf.nn.tanh(x)
    
def lrelu(name,x):
    with tf.variable_scope(name):
        return tf.nn.leaky_relu(x)
    
def pool(name,x):  #  Avoid calculating "padding zeros"
    shape = [i.value for i in x.get_shape()]
    ones = np.ones([1,shape[1],shape[2],1],np.float32)
    with tf.variable_scope(name):
        mask = tf.constant(ones,tf.float32,name='Mask')
        x_pool = tf.nn.avg_pool(x,[1,3,3,1],[1,2,2,1],"SAME")
        mask_pool = tf.nn.avg_pool(mask,[1,3,3,1],[1,2,2,1],"SAME")
        final_pool = x_pool/mask_pool
        return final_pool
