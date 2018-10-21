from scipy import misc
import numpy as np
import tensorflow as tf
import math
import functools
import os


def res_block(name,x):   # conv('conv1', x, 3*3, 64, 2, ref_pad=1, pad=True)
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,3*3,1024,1,1,False)
        conv1_ins = ins_norm('ins1',conv1)
        conv1_relu = relu('relu1',conv1_ins)
        
        conv2 = conv('conv2',conv1_relu,3*3,1024,1,1,False)
        conv2_ins = ins_norm('ins2',conv2)
        conv2_relu = relu('relu2',conv2_ins) 
        return conv2_relu+x


def G_base(name,x,batch):  
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,7*7,64,1,3,False)
        ins1 = ins_norm('ins1',conv1)
        relu1 = relu('relu1',ins1)
        
        conv2 = conv('conv2',relu1,3*3,128,2,0,True)
        ins2 = ins_norm('ins2',conv2)
        relu2 = relu('relu2',ins2)
        
        conv3 = conv('conv3',relu2,3*3,256,2,0,True)
        ins3 = ins_norm('ins3',conv3)
        relu3 = relu('relu3',ins3)
        
        conv4 = conv('conv4',relu3,3*3,512,2,0,True)
        ins4 = ins_norm('ins4',conv4)
        relu4 = relu('relu4',ins4)
        
        conv5 = conv('conv5',relu4,3*3,1024,2,0,True)
        ins5 = ins_norm('ins5',conv5)
        relu5 = relu('relu5',ins5)
        
        x_in = relu5
#         for i in range(9):
#             name = 'res'+str(i+1)
#             x_in = res_block(name,x_in)
        for i in range(9):
            name = 'res'+str(i+1)
            x_in = res_block(name,x_in)
            
        up1 = conv_trans('up1',x_in,3*3,512,2,batch,True)
        ins_up1 = ins_norm('ins_up1',up1)
        relu_up1 = relu('relu_up1',ins_up1)
        
        up2 = conv_trans('up2',relu_up1,3*3,256,2,batch,True)
        ins_up2 = ins_norm('ins_up2',up2)
        relu_up2 = relu('relu_up2',ins_up2)
        
        up3 = conv_trans('up3',relu_up2,3*3,128,2,batch,True) 
        ins_up3 = ins_norm('ins_up3',up3)
        relu_up3 = relu('relu_up3',ins_up3)
        
        up4 = conv_trans('up4',relu_up3,3*3,64,2,batch,True)
        ins_up4 = ins_norm('ins_up4',up4)
        relu_up4 = relu('relu_up4',ins_up4)
#                     conv('conv1', x, 3*3, 64, 2, ref_pad=1, pad=True) 
        conv_end = conv('conv_end',relu_up4,7*7,3,1,3,False)
        tanh_end = tanh('tanh_end',conv_end)
        return tanh_end

def D_base(name,x):
    with tf.variable_scope(name):
        conv1 = conv('conv1',x,4*4,64,2,0,True)
        l1 = lrelu('lrelu1',conv1)
        
        conv2 = conv('conv2',l1,4*4,128,2,0,True)
        ins2 = ins_norm('ins2',conv2)
        l2 = lrelu('lrelu2',ins2)
        
        conv3 = conv('conv3',l2,4*4,256,2,0,True)
        ins3 = ins_norm('ins3',conv3)
        l3 = lrelu('lrelu3',ins3)
        
        conv4 = conv('conv4',l3,4*4,512,1,0,True)
        ins4 = ins_norm('ins4',conv4)
        l4 = lrelu('lrelu4',ins4)
        
        conv5 = conv('conv5',l4,4*4,1,1,0,True)
        return [l1,l2,l3,l4,conv5]

def feat_loss(d1_r,d1_f,d2_r,d2_f,feat_weight,d_weight):
    feat_1,feat_2 = [],[]
    for i in range(len(d1_r)):
        l1_loss = tf.reduce_mean(tf.abs(d1_r[i]-d1_f[i]))*feat_weight*d_weight*(4/(1+5))
        feat_1.append(l1_loss)
    for i in range(len(d2_r)):
        l1_loss = tf.reduce_mean(tf.abs(d2_r[i]-d2_f[i]))*feat_weight*d_weight*(4/(1+5))
        feat_2.append(l1_loss)
    feat = feat_1 + feat_2
    loss_total = functools.reduce(tf.add,feat)
    return loss_total

def Save_im(ims,save_dir,ce,cb):
    im_norm = (ims+1)/2
    shape = np.shape(ims)
    for num in range(shape[0]):
        name = save_dir+'/ep_'+str(ce)+'_batch_'+str(cb)+'_num_'+str(num)+'.png'
        misc.imsave(name,im_norm[num,:,:,:])
        
def get_edge(im):
    edge = np.zeros_like(im)
    edge[:,1:] = edge[:,1:] | (im[:,1:] != im[:,:-1])
    edge[:,:-1] = edge[:,:-1] | (im[:,1:] != im[:,:-1])
    edge[1:,:] = edge[1:,:] | (im[1:,:] != im[:-1,:])
    edge[:-1,:] = edge[:-1,:] | (im[1:,:] != im[:-1,:])
    Edge = edge.astype(np.float32)
    return Edge
        
def load_data(label_dir,ins_dir,high,width):
    num_label = len(os.listdir(label_dir))
    num_inst = len(os.listdir(ins_dir))
    labels = np.zeros([num_label,width,high],np.int32)
    bounds = np.zeros([num_inst,width,high],np.float32)
    for idx,label_name in enumerate(os.listdir(label_dir)):
        label_im = misc.imread(label_dir+'/'+label_name).astype(np.int32)
        labels[idx,:,:] = label_im
    for idx,ins_name in enumerate(os.listdir(ins_dir)):
        ins_im = misc.imread(ins_dir+'/'+ins_name)
        bound = get_edge(ins_im)
        bounds[idx,:,:] = bound
    Labels = labels.astype(np.int32)
    Bounds = bounds.astype(np.float32)
    return Labels,Bounds
