# pix2pixhd_Tensorflow
Pix2pix primary architecture based solely on tensorflow
## Create tfrecord
&#8195;In order to speed up the reading of data, the data is first formatted into tfrecords format. In this case, Cityscapes is used.First extract the label of the dataset, take a picture, and generate a boundary map:<br>
&#8195;&#8195;&#8195;python ./data/pix2pixHD/get_data.py --data_dir="./data/dataset/cityscapes/train" -- save_dir="./data/pix2pix/data" --phase="train"<br>
&#8195;&#8195;Then generating the tfrecord file:&#8195;python --file_label_dir="./data/pix2pix/data/train/label" --TFRECORD_DIR="./data/pix2pix/data/train/" --phase="train" --epoch=1000<br>
&#8195;The batch of training was assigned in advance because when the data was read with 'tf.train.shuffle_batch', although the sampling order of the samples was random, there was no guarantee that each sample would appear in a training period, so use 'tf.train. Batch' reads data. If you are unsure of the trained batch, you can set a larger value (but not too large).<br>
## The difference between training and inference
&#8195;During training, the code of the real picture is used as a feature input to the global generator; after the training is finished, the output feature space of the encoder can be separately clustered to obtain a specific code of a certain feature (such as the texture of the road). : asphalt road or stone road, as described in the paper). In the test, you need to specify the feature information manually. This function is still in the process of perfection, but you can enter the 'b_fed' in ‘Load_model’ by entering pix2pixhd to implement manual input.<br>
## Implementation of feature selection:
&#8195;For the output of the encoder, add two control quantities, k, b. Output = output(encoder) * k + b. When training, k=1, b=0; when inference, k=0, b is a manually added feature value.<br>
## Train:
&#8195;&#8195;&#8195;python ./data/pix2pixHD/train_test.py --phase="train" --epoch=500 --batch=1 --n_class=34 --num_d=2 --save_iter=5 --decay_ep=10 --im_high=1024 --im_width=2048 --feat_weight=10 --old_lr=0.002 --decay_weight=20 --sace_ckpt_iter=2 --data_dir="./data/pix2pix/data" --tf_record_dir="./data/pix2pix/data/train/" --save_path="./data/train/Logs" --save_im_dir="./data/train/Logs" --ckpt_dir="./data/train/Logs" --label_dir="./data/train/Logs" --inst_dir="./data/train/Logs"<br>
At training time,the input of ckpt, label_dir, and ins_dir is not required during training, just for the setting of argparse.
