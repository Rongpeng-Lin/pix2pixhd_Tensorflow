from scipy import misc
import numpy as np
import tensorflow as tf
import math
import functools
import os


def conv(name,x,kers,outs,s,ref_pad,pad):
    shape = [i.value for i in x.get_shape()]
    ker = math.sqrt(kers)
    ins = int(kers*shape[-1])
    ins_min,ins_max = 1/math.sqrt(ins),(-1)/math.sqrt(ins)
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,shape[-1],outs],
                            tf.float32,
                            tf.random_uniform_initializer(ins_min,ins_max))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.random_uniform_initializer(ins_min,ins_max))
        if ref_pad:
            x_pad = tf.pad(x,[[0,0],[ref_pad,ref_pad],[ref_pad,ref_pad],[0,0]])
            return tf.nn.conv2d(x_pad,w,[1,s,s,1],"VALID") + b
        else:
            paded = "SAME" if pad else "VALID"
            return tf.nn.conv2d(x,w,[1,s,s,1],paded) + b
        
def conv_D(name,x,kers,outs,s,ref_pad,pad):
    shape = [i.value for i in x.get_shape()]
    ker = math.sqrt(kers)
    ins = int(kers*shape[-1])
    ins_min,ins_max = 1/math.sqrt(ins),(-1)/math.sqrt(ins)
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,shape[-1],outs],
                            tf.float32,
                            tf.initializers.truncated_normal(stddev=0.02))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.random_uniform_initializer(ins_min,ins_max))
        if ref_pad:
            x_pad = tf.pad(x,[[0,0],[ref_pad,ref_pad],[ref_pad,ref_pad],[0,0]])
            return tf.nn.conv2d(x_pad,w,[1,s,s,1],"VALID") + b
        else:
            paded = "SAME" if pad else "VALID"
            return tf.nn.conv2d(x,w,[1,s,s,1],paded) + b
        
def conv_trans(name,x,kers,outs,s,b,pad):  #  要给batch
    shape = [i.value for i in x.get_shape()]
    w,h,c = shape[1],shape[2],shape[3]
    ker = math.sqrt(kers)
    outshape = [b,int(s*w),int(s*h),outs]
    with tf.variable_scope(name):
        w = tf.get_variable('w',
                            [ker,ker,outs,c],
                            tf.float32,
                            tf.random_uniform_initializer(0,1))
        b = tf.get_variable('b',
                            [outs],
                            tf.float32,
                            tf.random_uniform_initializer(0,1))
        paded = "SAME" if pad else "VALID"
        return tf.nn.conv2d_transpose(x,w,outshape,[1,s,s,1],padding=paded) + b
