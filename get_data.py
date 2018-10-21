import os
import scipy.misc as misc
import numpy as np
import shutil
import argparse,sys


def get_value(L):   # Calculate the center value
    center = L[1,1]
    up = L[0,1]
    down = L[2,1]
    left = L[1,0]
    right = L[1,2]
    s = up+down+left+right-4*center
    if s!=0:
        v = 1.
    else:
        v = 0.
    return v

def contorous(labeldir,save_dir):        # Draw a large outline
    k = np.zeros([1024,2048])
    label = misc.imread(labeldir)
    shape = np.shape(label)
    for i in range(1,shape[0]-1):
        for j in range(1,shape[1]-1):
            local = label[(i-1):(i+2),(j-1):(j+2)]
            value = get_value(local)
            k[i,j] = value
    k1 = k.astype(np.float32)
    misc.imsave(save_dir,k1)
    return True

def main(args):
    data_dir = args.data_dir
    save_dir = args.save_dir
    phase = args.phase
    raw_data_file = data_dir + '/'+phase+'/'
    raw_data_real = data_dir + '/leftImg8bit/'+ phase

    train_file = save_dir+'/'+phase
    labels_dir = train_file+'/'+'label'
    bounds_dir = train_file+'/'+'bound'
    real_im = train_file+'/'+'Realpic'

    if not os.path.exists(train_file):
        os.makedirs(train_file)
        os.makedirs(labels_dir)
        os.makedirs(bounds_dir)
        os.makedirs(real_im)
   #    get bound maps and label maps 
    total = 0
    for file in os.listdir(raw_data_file):
        for im_name in os.listdir(raw_data_file+file):
            im_type = im_name.split('_')[-1]
            if im_type=='instanceIds.png':
                im_dir = raw_data_file+file+'/'+im_name
                contorous(im_dir,bounds_dir+'/'+im_name)
            if im_type=='labelIds.png':
                im_dir = raw_data_file+file+'/'+im_name
                shutil.copy(im_dir,labels_dir+'/'+im_name)
            total+=1
    print('finish get images: ',total)
   #   get real images
    total = 0
    for file in os.listdir(raw_data_real):
        for im_name in os.listdir(raw_data_real+'/'+file):
            im_dir = raw_data_real+'/'+file+'/'+im_name
            shutil.copy(im_dir,real_im)
            total+=1
    print('finish get real images: ',total)
    return True

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Raw images dir.', default="./rawdata")   
    parser.add_argument('--save_dir', type=str, help='Images save dir.', default="./data")
    parser.add_argument('--phase', type=str, help='train/test.', default='train')
    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))

