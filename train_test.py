import argparse,sys
from pix2pixhd import *


def parse_arguments(opts):
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, help='train or test', default="train")
    parser.add_argument('--epoch', type=int, help='Epoch', default=300)
    parser.add_argument('--batch', type=int, help='Batch_size', default=1)
    parser.add_argument('--n_class', type=int, help='The number of categories represented by the label', default=34)
    parser.add_argument('--num_d', type=int, help='Number of discriminators', default=2)
    parser.add_argument('--save_iter', type=int, help='Number of intervals to save the model', default=5)
    parser.add_argument('--decay_ep', type=int, help='Attenuation control coefficient', default=40)
    parser.add_argument('--im_high', type=int, help='The height of the image', default=1024)
    parser.add_argument('--im_width', type=int, help='The width of the image', default=2048)
    parser.add_argument('--feat_weight', type=float, help='Weight of feature loss', default=2)
    parser.add_argument('--old_lr', type=float, help='Initial learning rate', default=2)
    parser.add_argument('--decay_weight', type=int, help='Rate of decay of weight', default=2)
    parser.add_argument('--sace_ckpt_iter', type=int, help='Number of cycles to save the model', default=5)
    parser.add_argument('--data_dir', type=str, help='Path to label/instance image data', default="./data/train/label")
    parser.add_argument('--tf_record_dir', type=str, help='Path to the tfrecord file', default="./data/train/train.tfrecords")
    parser.add_argument('--save_path', type=str, help='Model save paths', default="./data/train/Logs")
    parser.add_argument('--save_im_dir', type=str, help='Image save paths', default="./data/train/Logs")
    parser.add_argument('--ckpt_dir', type=str, help='Model load dir', default="./data/train/Logs/model.ckpt-200")
    parser.add_argument('--label_dir', type=str, help='Label path during testing', default="./data/model/Label")
    parser.add_argument('--inst_dir', type=str, help='Label path during testing', default="./data/model/Label")
    return parser.parse_args(argv)

def main(opt):
    HD = pix2pixHD(opt)
    if opt.phase=='train':
        HD.train()
    else:
#         b_fed : your features producted by encoder after clustering 
        HD.Load_model(b_fed)
    return True         

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
