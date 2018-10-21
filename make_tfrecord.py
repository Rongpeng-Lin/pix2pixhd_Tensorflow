import argparse,sys
import tensorflow as tf
import os
import random
import numpy as np
import scipy.misc as misc


def all_dirs(file_dir):   
    dirs = []
    for im_name in os.listdir(file_dir):
        dirs.append(file_dir+'/'+im_name)
    return dirs

# There are some differences from the official example. 
# If you call tf.train.shuffle_batch() directly, the number of images may be inconsistent.
def shuffle(names,ep):   
    Names = names
    retu = []
    for i in range(ep):
        random.shuffle(Names)
        retu += Names
    return retu

def find_im(filename,feature):
    im_name = filename.split('/')[-1]
    if feature=='bound':
        name_list = im_name.split('_')[:-1]
        new_name = '_'.join(name_list+['instanceIds']) + '.png'
        name_l = filename.split('/')[:-1]
        name_l[-1] = 'bound'
    else:
        name_list = im_name.split('_')[:-2]
        new_name = '_'.join(name_list+['leftImg8bit']) + '.png'
        name_l = filename.split('/')[:-1]
        name_l[-1] = 'Realpic'
    return '/'.join(name_l+[new_name])
        
    
def bytes_feature(values):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def image_to_tfexample(label,real,bound):
    return tf.train.Example(features=tf.train.Features(feature={
                'Label':bytes_feature(label),
                'Real':bytes_feature(real),
                'Bound':bytes_feature(bound)}))

def _convert_dataset(split_name, filenames, TFRECORD_DIR):
    assert split_name in ['train', 'test']
    with tf.Session() as sess:                                
        output_filename = os.path.join(TFRECORD_DIR,split_name + '.tfrecords')
        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            for i,label_name in enumerate(filenames):
                bound_name = find_im(label_name,'bound')
                real_name = find_im(label_name,'real')
                
                label_data = misc.imread(label_name)
                real_data = misc.imread(real_name)
                bound_data = misc.imread(bound_name)
                bound_data = (bound_data/255).astype(np.uint8)
                
                label_data = label_data.tobytes()  
                real_data = real_data.tobytes() 
                bound_data = bound_data.tobytes()
                                               
                example = image_to_tfexample(label_data,real_data,bound_data)
                tfrecord_writer.write(example.SerializeToString())
                print('已处理图片: ',i)
                
def main(args):
    print(args.file_label_dir)
    filenames = all_dirs(args.file_label_dir)
    training_filenames = shuffle(filenames,args.epoch)
    _convert_dataset(args.phase, training_filenames, args.TFRECORD_DIR)
    return True


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_label_dir', type=str, help='Train or test label file dir.', default="./data/train/label")   
    parser.add_argument('--TFRECORD_DIR', type=str, help='The save path of the tfrecord file.', default="./data/train/")
    parser.add_argument('--phase', type=str, help='Train or test record.', default="train")
    parser.add_argument('--epoch', type=int, help='The number of cycles to train, if not sure, can be set to a large value.', default=10000)
    return parser.parse_args(argv)
    

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
