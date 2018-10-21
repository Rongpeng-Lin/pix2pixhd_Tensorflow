from activations import *
from conv_base import *
from blocks import *

class pix2pixHD:
    def __init__(self,opt):
        self.epoch = opt.epoch
        self.batch = opt.batch
        self.n_class = opt.n_class
        self.d_weight = 1/opt.num_d
        self.feat_weight = opt.feat_weight
        self.old_lr = opt.old_lr
        self.save_iter = opt.save_iter
        self.decay_ep = opt.decay_ep
        self.decay_weight = opt.decay_weight
        self.im_width = opt.im_width
        self.im_high = opt.im_high
        self.sace_ckpt_iter = opt.sace_ckpt_iter
        self.n_im = len(os.listdir(opt.data_dir))
        
        self.tf_record_dir = opt.tf_record_dir
        self.save_path = opt.save_path    #'./data/pix2pixhd/Logs'
        self.save_im_dir = opt.save_im_dir  
        self.ckpt_dir = opt.ckpt_dir    #  './data/pix2pixhd/Logs/model.ckpt-10'
        self.label_dir = opt.label_dir
        self.inst_dir = opt.inst_dir
        
        self.label = tf.placeholder(tf.int32,[None,self.im_width,self.im_high])
        self.bound = tf.placeholder(tf.float32,[None,self.im_width,self.im_high])
        self.real_im = tf.placeholder(tf.float32,[None,self.im_width,self.im_high,3])
        self.k = tf.placeholder(tf.float32,[1])
        self.b = tf.placeholder(tf.float32,[None,self.im_width,self.im_high,3])
        # process
        self.onehot = tf.one_hot(self.label,self.n_class)
        self.bound_ = tf.expand_dims(self.bound,3)
        self.real_im_ = self.real_im/255
        
    #############################  data_loader ##################################
    def read_and_decode(self,filename):
        filename_queue = tf.train.string_input_producer([filename])
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)   
        features = tf.parse_single_example(serialized_example,
                                           features={
                                               'Label':tf.FixedLenFeature([], tf.string),
                                               'Real':tf.FixedLenFeature([], tf.string),
                                               'Bound':tf.FixedLenFeature([], tf.string),
                                           })
        image_label = tf.decode_raw(features['Label'], tf.uint8)
        image_label = tf.reshape(image_label, [1024, 2048])

        image_real = tf.decode_raw(features['Real'], tf.uint8)
        image_real = tf.reshape(image_real, [1024, 2048, 3])

        image_bound = tf.decode_raw(features['Bound'], tf.uint8)
        image_bound = tf.reshape(image_bound, [1024, 2048])
        return image_label,image_real,image_bound
     ###############################################################################
    
    def build_G(self,x_bound,x_label,x_feat,x_k,x_b):
        with tf.variable_scope('G_net'):
            x_feat_act = tf.add(tf.multiply(x_feat,x_k),x_b)
            x_concat = tf.concat([x_bound,x_label,x_feat_act],3)
            conv1 = conv('conv1',x_concat,3*3,3,1,1,False)
            conv2 = conv('conv2',conv1,3*3,3,1,1,False)
            conv3 = conv('conv3',conv2,3*3,3,2,1,False)
            x_pool = pool('pool_x',x_concat)
            G1_tanh_end = G_base('G1',x_pool,self.batch)
            G_add = tf.add(G1_tanh_end,conv3,name='G_Add')
            conv4 = conv('conv4',G_add,3*3,1024,1,1,False)
            res_1 = res_block('res_1',conv4)
            res_2 = res_block('res_2',res_1)
            trans1 = conv_trans('trans1',res_2,3*3,3,2,self.batch,True)
            trans_tanh = tanh('trans_tanh',trans1)
            return trans_tanh
        
    def build_D1(self,im,label,reuse):
        with tf.variable_scope('D1',reuse=reuse):
            x_ = tf.concat([im,label],3)
            D = D_base('D',x_)         
            return D

    def build_D2(self,im,label,reuse):
        with tf.variable_scope('D2',reuse=reuse):
            x_ = tf.concat([im,label],3)
            x_pool = pool('pool_D',x_) 
            D = D_base('D',x_pool)         
            return D
        
    def encoder(self,x):
        with tf.variable_scope('Encoder'):
            x_encode = G_base('encode',x,self.batch)
            return x_encode
        
    def forward(self):
        self.x_feat = self.encoder(self.real_im_)
        
        self.fake_im = self.build_G(self.bound_,self.onehot,self.x_feat,self.k,self.b)
        self.real_D1_out = self.build_D1(self.real_im_,self.onehot,False)
        self.fake_D1_out = self.build_D1(self.fake_im,self.onehot,True)
        
        self.real_D2_out = self.build_D2(self.real_im_,self.onehot,False)
        self.fake_D2_out = self.build_D2(self.fake_im,self.onehot,True)
       
    def cacu_loss(self):
        self.lsgan_d1 = tf.reduce_mean(0.5*tf.square(self.real_D1_out[-1]-1) + 0.5*tf.square(self.fake_D1_out[-1]))                    
        self.lsgan_d2 = tf.reduce_mean(0.5*tf.square(self.real_D2_out[-1]-1) + 0.5*tf.square(self.fake_D2_out[-1]))
        self.lsgan_g = 0.5*tf.reduce_mean(tf.square(self.fake_D2_out[-1]-1)) + 0.5*tf.reduce_mean(tf.square(self.fake_D1_out[-1]-1))
        self.feat_loss = feat_loss(self.real_D1_out, self.fake_D1_out, self.real_D2_out, self.fake_D2_out, self.feat_weight, self.d_weight)
        tf.summary.scalar('d1_loss',self.lsgan_d1)
        tf.summary.scalar('d2_loss',self.lsgan_d2)
        tf.summary.scalar('g_loss',self.lsgan_g)
        tf.summary.scalar('feat_loss',self.feat_loss)
        
    def train(self):
        lr = self.old_lr
        self.forward()
        self.cacu_loss()
        D1_vars = [var for var in tf.all_variables() if 'D1' in var.name]
        D2_vars = [var for var in tf.all_variables() if 'D2' in var.name]
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        encoder_vars = [var for var in tf.all_variables() if 'Encoder' in var.name]
        optim_D1 = tf.train.AdamOptimizer(lr).minimize(self.lsgan_d1,var_list=D1_vars)
        optim_D2 = tf.train.AdamOptimizer(lr).minimize(self.lsgan_d2,var_list=D2_vars)
        optim_G_ALL = tf.train.AdamOptimizer(lr).minimize(self.lsgan_g+self.feat_loss,var_list=G_vars+encoder_vars)
        
        im_l,im_re,im_bound = read_and_decode(self.tf_record_dir)
        label_batch,real_batch,bound_batch = tf.train.batch(
            [im_l,im_re,im_bound],
            batch_size = self.batch,
            capacity = 50000)
        
        with tf.Session() as sess:
            k_fed = np.ones([1],np.float32)
            b_fed = np.zeros([self.batch,self.im_width,self.im_high,3],np.float32)
            
            sess.run(tf.global_variables_initializer())
            merge = tf.summary.merge_all()
            graph = tf.summary.FileWriter(self.save_path, sess.graph)
            Saver = tf.train.Saver(max_to_keep=10)
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess, coord=coord)
            for ep in range(self.epoch):
                for j in range(self.n_im//self.batch):                    

                    label_fed,real_im_fed,bound_fed = sess.run([label_batch,real_batch,bound_batch])
                    dict_ = {self.label:label_fed,self.bound:bound_fed,self.real_im:real_im_fed,self.k:k_fed,self.b:b_fed}
                    step = int(ep*(self.n_im//self.batch)+j)
                    _,_ = sess.run([optim_D1,optim_D2],feed_dict=dict_)
                    _,fake_im,Merge = sess.run([optim_G_ALL,self.fake_im,merge],feed_dict=dict_)
                    graph.add_summary(Merge,step)
                    if (ep*self.n_im+j*self.batch)//self.save_iter==0:
                        Save_im(fake_im,self.save_im_dir,ep,j)
                    if (j*self.batch+ep*self.n_im)%self.sace_ckpt_iter==0:
                        num_trained = int(j*self.batch+ep*self.n_im)
                        Saver.save(sess,self.save_path+'/'+'model.ckpt',num_trained)
                        print('save success at num images trained: ',num_trained)
                if ep>self.decay_ep:
                    lr = self.old_lr - ep/self.decay_weight
            coord.request_stop()
            coord.join(threads)
            return True
        
    def Load_model(self,b_fed):
        #  b_fed is a feature vector extracted from the encoder's encoding and needs to be specified 
        #   by human (by clustering the results of the trained encoder).
        self.x_feat = self.encoder(self.real_im_)
        self.fake_im = self.build_G(self.bound_,self.onehot,self.x_feat,self.k,self.b)       
        G_vars = [var for var in tf.all_variables() if 'G' in var.name]
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            graph = tf.summary.FileWriter(self.logdir, sess.graph)
            Saver = tf.train.Saver(var_list=G_vars)
            Saver.restore(sess,self.ckpt_dir)
            
            label_fed,bound_fed = load_data(self.label_dir,self.inst_dir)
            #  k_fed must be zero, which means that the actual output of the encoder is not considered, because there is no ideal result color map when used. 
            #      (The characteristic input of G is: output(encoder)*k+b, k=1 during training, b=0)
            k_fed = np.zeros([1],np.float32)
            
            real_im_fed = np.zeros([np.shape(label_fed)[0],self.im_width,self.im_high,3],np.float32)
            
            dict_ = {self.label:label_fed,self.bound:bound_fed,self.real_im:real_im_fed,self.k:k_fed,self.b:b_fed}
            
            ims = sess.run(self.fake_im,feed_dict=dict_)
            Save_im(ims,self.save_im_dir,0,0)
            print(np.shape(ims))
