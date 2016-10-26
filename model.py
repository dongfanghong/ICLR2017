import os
import time
from glob import glob
import tensorflow as tf
import pickle
from ops import *
from utils import *
import numpy as np

samples_dir = '/atlas/u/dfh13/samples_new2'
eval_dir = '/atlas/u/dfh13/eval_new2'
log_dir = '/atlas/u/dfh13/logs_new2'

for path in [samples_dir,eval_dir,log_dir]:
    try: 
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise

class DCGAN(object):
    def __init__(self, sess, image_size=108, is_crop=True,
                 batch_size=36, sample_size = 64, image_shape=[64, 64, 3],
                 y_dim=None, z_dim=100, gf_dim=64, df_dim=64,
                 gfc_dim=1024, dfc_dim=1024, c_dim=3, dataset_name='default',
                 checkpoint_dir=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen untis for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. [3]
        """
        self.sess = sess
        self.is_crop = is_crop
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.image_shape = image_shape
        self.dataset_name = dataset_name

        self.validation_size = self.batch_size * 30

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.gfc_dim = gfc_dim
        self.dfc_dim = dfc_dim

        self.c_dim = 3

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')

        if not self.y_dim:
            self.d_bn3 = batch_norm(name='d_bn3')

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')

        if not self.y_dim:
            self.g_bn3 = batch_norm(name='g_bn3')

        self.e_bn1 = batch_norm(name='e_bn1')
        self.e_bn1_1 = batch_norm(name='e_bn1_1')
        self.e_bn1_2 = batch_norm(name='e_bn1_2')
        self.e_bn2 = batch_norm(name='e_bn2')
        self.e_bn3 = batch_norm(name='e_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = checkpoint_dir

        self.first_encoder = True
        self.first_encoder2 = True

        self.build_model()

    def build_model(self):
        if self.y_dim:
            self.y= tf.placeholder(tf.float32, [None, self.y_dim], name='y')
        '''
        self.images = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape,
                                    name='real_images')
        self.sample_images= tf.placeholder(tf.float32, [self.sample_size] + self.image_shape,
                                        name='sample_images')
        '''
        self.patch1 = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='patch1')
        self.patch2 = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='patch2')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.patch1, self.z)
        
        self.D, self.D_logits = self.discriminator(self.patch1, self.patch2)
        
        self.sampler = self.sampler(self.patch1, self.z)
        self.D_, self.D_logits_ = self.discriminator(self.patch1, self.G, reuse=True)

        self.d_sum = tf.histogram_summary("d", self.D)
        self.d__sum = tf.histogram_summary("d_", self.D_)
        self.G_sum = tf.image_summary("G", self.G)

        self.d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits, tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.zeros_like(self.D_)))
        self.g_loss_adv = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(self.D_logits_, tf.ones_like(self.D_)))
        self.l1_error = tf.abs(self.G-self.patch2)#*self.mask
        self.error_sum = tf.image_summary("l1_error", self.l1_error)
        #print self.l1_error.get_shape()
        self.g_loss_l1 = tf.reduce_mean(self.l1_error)#/tf.reduce_mean(self.mask)

        self.d_loss_real_sum = tf.scalar_summary("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.scalar_summary("d_loss_fake", self.d_loss_fake)
        self.g_loss_adv_sum = tf.scalar_summary("g_loss_adv", self.g_loss_adv)
        self.g_loss_l1_sum = tf.scalar_summary("g_loss_l1", self.g_loss_l1)
                                                    
        self.d_loss = self.d_loss_real + self.d_loss_fake
        self.g_loss = 0.01 * self.g_loss_adv + self.g_loss_l1
        #self.g_loss = self.g_loss_l1

        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)
        self.d_loss_sum = tf.scalar_summary("d_loss", self.d_loss)
   
        t_vars = tf.trainable_variables()
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""

        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(0.25*config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim_l1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss_l1, var_list=self.g_vars)
        g_optim_adv = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss_adv, var_list=self.g_vars)

        g_optim_tot = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()

        self.sum = tf.merge_all_summaries()

        self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph)

        if False and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        image_list_filename = '/atlas/u/dfh13/faces/face_list.pkl'
        print image_list_filename
        #image_list_filename = '/atlas/u/dfh13/bedroom_list.pkl'
        if (os.path.isfile(image_list_filename)):
            self.data_all = pickle.load(open(image_list_filename,'rb'))
            print len(self.data_all)
        else:
            print "Start finding images"
            start = time.time()
            self.data_all = find_files("/atlas/u/dfh13/faces/","*.jpg")
            pickle.dump(self.data_all,open(image_list_filename,'wb'))
            print "Finish finding images in", time.time()-start, 's, find',len(self.data_all),'images'

        def train_loop():
            sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

            self.validation_files = self.data_all[len(self.data_all)-self.validation_size:len(self.data_all)]
            self.sample_files = self.validation_files[:self.batch_size]

            sample_patch1, sample_patch2 = get_patches_batch(self.sample_files,get_patches)

            #sample_images = np.array(sample).astype(np.float32)
            data = self.data_all[:len(self.data_all)-self.validation_size]
            counter = 1
            start_time = time.time()
       
            for epoch in xrange(config.epoch):
                #data = glob(os.path.join("./data", config.dataset, "*.JPEG"))
                batch_idxs = min(len(data), config.train_size)/config.batch_size
    
                for idx in xrange(0, batch_idxs):
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    #batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                    #print "Start loading"
                    #start = time.time()

                    g_optim = g_optim_tot

                    patch1_batch, patch2_batch = get_patches_batch(batch_files,get_patches)
    
                    #print 'load time:', time.time()-start
                    #batch_images = np.array(batch).astype(np.float32)
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
    
                    start = time.time()
                    errD_fake, errD_real, errG_adv, errG_l1, summary_str = self.sess.run([self.d_loss_fake, self.d_loss_real, self.g_loss_adv, self.g_loss_l1, self.sum],
                        feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)
    
                    flag = errG_l1 < 0.1
                    #print 'eval time:', time.time()-start
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                        % (epoch, idx, batch_idxs,time.time() - start_time), errD_fake, errD_real, errG_adv, errG_l1)
    
                    
                    errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, g_optim],
                        feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z})
                    #self.writer.add_summary(summary_str, counter)
                    
                    if True:
                        errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, d_optim],
                            feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z})
                        #self.writer.add_summary(summary_str, counter)
                                               
                        if errD_fake > errG_adv:
                            cnt = 0
                            while errD_fake > errG_adv:
                                errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, d_optim],
                                    feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z})
                                cnt += 1
                                if cnt > 1:
                                    break
                        else:
                            cnt = 0
                            while errG_adv > errD_fake:
                                errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, g_optim],
                                    feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z})
                                cnt += 1
                                if cnt > 1:
                                    break
                        
        
                    #errD_fake = self.d_loss_fake.eval({self.patch1: patch1_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    #errD_real = self.d_loss_real.eval({self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    #errG = self.g_loss.eval({self.patch1: patch1_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
    
                    if np.mod(counter, 30) == 1:
                        print "Generating samples"
                        samples, loss_l1 = self.sess.run(
                            [self.sampler, self.g_loss_l1],
                            feed_dict={ self.patch1: sample_patch1, self.patch2: sample_patch2, self.z: batch_z}
                        )
                        save_images(samples, [6, 6],
                                    samples_dir+'/test_%s_%s_synthesized.png' % (epoch, idx))
                        save_images(sample_patch1, [6, 6],
                                    samples_dir+'/test_%s_%s_org.png' % (epoch, idx))
                        save_images(sample_patch2, [6, 6],
                                    samples_dir+'/test_%s_%s_target.png' % (epoch, idx))                    
                        print("[Sample] g_loss_l1: %.8f" % (loss_l1))
    
                    if np.mod(counter, 30) == 1:
                        print "Training results"
                        samples, loss_l1 = self.sess.run(
                            [self.sampler, self.g_loss_l1],
                            feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.z: batch_z}
                        )
                        save_images(samples, [6, 6],
                                    samples_dir+'/train_%s_%s_synthesized.png' % (epoch, idx))
                        save_images(patch1_batch, [6, 6],
                                    samples_dir+'/train_%s_%s_org.png' % (epoch, idx))
                        save_images(patch2_batch, [6, 6],
                                    samples_dir+'/train_%s_%s_target.png' % (epoch, idx))                    
                        print("[Sample] g_loss_l1: %.8f" % (loss_l1))                    
    
                    #if counter == 10:
                        #self.interpolate()
                        #xxx
                    #if np.mod(counter, 100) == 1:
                        #self.test3(epoch,idx)
                    #if np.mod(counter, 100) == 1:
                        #self.test2(epoch,idx)
                    if np.mod(counter, 3000) == 0:
                        print "Saving checkpoint"
                        self.save(config.checkpoint_dir, counter)
                        print "Save finished"
 
        train_loop()
        self.retrieve()

    def discriminator(self, patch1, patch2, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if True:
            #encoded_patch1 = self.encoder(patch1)
            encoded_patch2 = self.encoder(patch2)
            joint = encoded_patch2
            #joint = tf.concat(3,[encoded_patch1,encoded_patch2])

            h3 = joint

            h3_reshape = tf.reshape(h3, [self.batch_size, -1])

            h4 = linear(h3_reshape, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, patch1, z, y=None):
        if True:
            # project `z` and reshape
            encoded_patch1 = self.encoder2(patch1)
            
            self.z_ = linear(tf.concat(1,[z]), self.gf_dim*8*4*4, 'g_h0_lin')

            self.h0_z = tf.reshape(self.z_, [self.batch_size, 4, 4, self.gf_dim * 8])
            #print encoded_patch1.get_shape(),self.h0_z.get_shape(),square_dx.get_shape()
            h0 = tf.concat(3,[self.h0_z,encoded_patch1])
            
            h0 = tf.nn.relu(self.g_bn0(h0))

            self.h1, self.h1_w, self.h1_b = deconv2d(h0, 
                [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1', with_w=True)
            h1 = tf.nn.relu(self.g_bn1(self.h1))

            h2, self.h2_w, self.h2_b = deconv2d(h1,
                [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2', with_w=True)
            h2 = tf.nn.relu(self.g_bn2(h2))

            h3, self.h3_w, self.h3_b = deconv2d(h2,
                [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3', with_w=True)
            h3 = tf.nn.relu(self.g_bn3(h3))

            h4, self.h4_w, self.h4_b = deconv2d(h3,
                [self.batch_size, 64, 64, 3], name='g_h4', with_w=True)

            return tf.nn.tanh(h4)

    def sampler(self, patch1, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if True:
            # project `z` and reshape
            encoded_patch1 = self.encoder2(patch1)

            h0_z = tf.reshape(linear(tf.concat(1,[z]), self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])

 
            h0 = tf.concat(3,[h0_z,encoded_patch1])
            
            h0 = tf.nn.relu(self.g_bn0(h0, train=False))

            h1 = deconv2d(h0, [self.batch_size, 8, 8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(self.g_bn1(h1, train=False))

            h2 = deconv2d(h1, [self.batch_size, 16, 16, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(self.g_bn2(h2, train=False))

            h3 = deconv2d(h2, [self.batch_size, 32, 32, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(self.g_bn3(h3, train=False))

            h4 = deconv2d(h3, [self.batch_size, 64, 64, 3], name='g_h4')

            return tf.nn.tanh(h4)

    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s" % (self.dataset_name, self.batch_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def encoder(self, image):
        with tf.variable_scope('encoder'):
            if self.first_encoder:
                self.first_encoder = False
            else:
                tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.df_dim,name='d_e_h0_conv'))
            h1 = tf.nn.relu(self.e_bn1(conv2d(h0,self.df_dim*2,name='d_e_h1_conv')))
            h2 = tf.nn.relu(self.e_bn2(conv2d(h1,self.df_dim*4,name='d_e_h2_conv')))
            h3 = tf.nn.relu(self.e_bn3(conv2d(h2,self.df_dim*8,name='d_e_h3_conv')))
    
            return h3

    def encoder_test(self, image):
        with tf.variable_scope('encoder'):
            tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.df_dim,name='d_e_h0_conv'))
            h1 = tf.nn.relu(self.e_bn1(conv2d(h0,self.df_dim*2,name='d_e_h1_conv'), train=False))
            h2 = tf.nn.relu(self.e_bn2(conv2d(h1,self.df_dim*4,name='d_e_h2_conv'), train=False))
            h3 = tf.nn.relu(self.e_bn3(conv2d(h2,self.df_dim*8,name='d_e_h3_conv'), train=False))
    
            return h3

    def encoder2(self, image):
        with tf.variable_scope('encoder2'):
            if self.first_encoder2:
                self.first_encoder2 = False
            else:
                tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.gf_dim,name='g_e_h0_conv'))
            h1 = tf.nn.relu(self.e_bn1(conv2d(h0,self.gf_dim*2,name='g_e_h1_conv')))
            #h1_1 = tf.nn.relu(self.e_bn1_1(conv2d(h1,self.gf_dim*4,d_h=1,d_w=1,name='g_e_h1_1_conv')))
            #h1_2 = tf.nn.relu(self.e_bn1_2(conv2d(h1_1,self.gf_dim*4,d_h=1,d_w=1,name='g_e_h1_2_conv')))
            h2 = tf.nn.relu(self.e_bn2(conv2d(h1,self.gf_dim*4,name='g_e_h2_conv')))
            h3 = tf.nn.relu(self.e_bn3(conv2d(h2,self.gf_dim*8,name='g_e_h3_conv')))
    
            return h3


    def interpolate(self):
        validation_files = self.data_all[len(self.data_all)-self.validation_size:len(self.data_all)]
        sample_files = validation_files[:self.batch_size]
        sample_patch1, sample_patch2 = get_patches_batch(self.sample_files,get_patches)

        #batch_z0 = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        #batch_z1 = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        batch_z0 = -np.ones([self.batch_size, self.z_dim],dtype='float32')
        batch_z1 = np.ones([self.batch_size, self.z_dim],dtype='float32')
        print batch_z0-batch_z1
        n = 15
        for i in xrange(n+1):
            print (i+0.0)/n,(n-i+0.0)/n
            batch_z = (i+0.0)/n*batch_z0+(n-i+0.0)/n*batch_z1
            samples, loss_l1 = self.sess.run(
                [self.sampler, self.g_loss_l1],
                feed_dict={ self.patch1: sample_patch1, self.patch2: sample_patch2, self.z: batch_z}
            )
            print loss_l1
            save_images(samples, [6, 6],
                eval_dir+'/eval_%s.png' % (i))

    def retrieve(self):
        print 'Running retrieval task'
        psize = 120

        feature = self.encoder_test(self.patch2)
        print feature.get_shape()
        feature_pooled = tf.nn.max_pool(feature, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
        print feature_pooled.get_shape()

        feat_list = []

        batch_idxs = len(self.data_all)/self.batch_size
        for idx in xrange(batch_idxs):
            print idx
            batch_files = self.data_all[idx*self.batch_size:(idx+1)*self.batch_size]
            img_batch = [get_image(filename, psize, is_crop=True) for filename in batch_files]
            img_batch = np.array(img_batch).astype(np.float32)
            feat = self.sess.run([feature_pooled],feed_dict={self.patch2:img_batch})
            feat = np.squeeze(feat)
            feat_list.extend(feat)
            #img_list.extend(batch_files)

        #feat_list = np.array(feat_list)
        #np.save('/atlas/u/dfh13/feat_database',feat_list)
        print 'Dumping result'
        pickle.dump(feat_list,open('/atlas/u/dfh13/'+self.dataset_name+'_feat_database.pkl','wb'))

