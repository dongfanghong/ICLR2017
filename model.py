import os
import time
from glob import glob
import tensorflow as tf
import pickle
from ops import *
from utils import *
import numpy as np

psize = 48
#samples_dir = '/atlas/u/dfh13/samples_'+str(psize)
suffix = 'tot_'
samples_dir = '/atlas/u/dfh13/latest/samples_'+suffix+str(psize)
eval_dir = '/atlas/u/dfh13/latest/eval_'+suffix+str(psize)
log_dir = '/atlas/u/dfh13/latest/logs_'+suffix+str(psize)
global_checkpoint_dir = '/atlas/u/dfh13/latest/checkpoint_'+suffix+str(psize)
feat_dir = '/atlas/u/dfh13/latest/feat_'+suffix+str(psize)

for path in [samples_dir,eval_dir,log_dir,global_checkpoint_dir,feat_dir]:
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

        self.d_e_bn1 = batch_norm(name='d_e_bn1')
        self.d_e_bn2 = batch_norm(name='d_e_bn2')
        self.d_e_bn3 = batch_norm(name='d_e_bn3')

        self.d_e2_bn1 = batch_norm(name='d_e2_bn1')
        self.d_e2_bn2 = batch_norm(name='d_e2_bn2')
        self.d_e2_bn3 = batch_norm(name='d_e2_bn3')        


        self.g_e_bn1 = batch_norm(name='g_e_bn1')
        self.g_e_bn2 = batch_norm(name='g_e_bn2')
        self.g_e_bn3 = batch_norm(name='g_e_bn3')

        self.dataset_name = dataset_name
        self.checkpoint_dir = global_checkpoint_dir
        #self.checkpoint_dir = checkpoint_dir

        self.first_d_encoder = True
        self.first_g_encoder = True

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
        self.dx = tf.placeholder(tf.float32, [self.batch_size], name='dx')
        self.dy = tf.placeholder(tf.float32, [self.batch_size], name='dy')
        self.mask = tf.placeholder(tf.float32, [self.batch_size] + self.image_shape, name='mask')

        self.z = tf.placeholder(tf.float32, [self.batch_size, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)

        self.G = self.generator(self.patch1, self.dx, self.dy, self.z)
        
        self.D, self.D_logits = self.discriminator(self.patch1, self.patch2, self.dx, self.dy)
        
        self.sampler = self.sampler(self.patch1,self.dx,self.dy,self.z)
        self.D_, self.D_logits_ = self.discriminator(self.patch1, self.G, self.dx, self.dy, reuse=True)

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

        self.patch1_uint8 = tf.cast((self.patch1+1)*127.5,tf.uint8)
        self.patch1_sum = tf.image_summary("patch1",self.patch1_uint8)
        self.patch2_uint8 = tf.cast((self.patch2+1)*127.5,tf.uint8)
        self.patch2_sum = tf.image_summary("patch2",self.patch2_uint8)        
        #self.patch2_sum = tf.image_summary("patch2",self.patch2)
        #self.G_sum = tf.image_summary("generated",self.G)

        t_vars = tf.trainable_variables()
        print [var.name for var in t_vars]
        #print [var.name for var in tf.all_variables()]
        
        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()

    def train(self, config):
        """Train DCGAN"""

        #np.random.shuffle(data)

        d_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.d_loss, var_list=self.d_vars)
        g_optim_l1 = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss_l1, var_list=self.g_vars)
        g_optim_adv = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss_adv, var_list=self.g_vars)

        g_optim_tot = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        '''
        temp = set(tf.all_variables())
        
        sess.run(tf.initialize_variables(set(tf.all_variables()) - temp))
        '''

        tf.initialize_all_variables().run()

        self.saver = tf.train.Saver()
        '''
        self.g_sum = tf.merge_summary([self.z_sum, self.d__sum, 
            self.G_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.merge_summary([self.z_sum, self.d_sum, 
            self.d_loss_real_sum, self.d_loss_sum])
        '''
        self.sum = tf.merge_all_summaries()

        self.writer = tf.train.SummaryWriter(log_dir, self.sess.graph)
        print 'here'
        if True and self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")

        image_list_filename = 'face_list.pkl'
        #image_list_filename = '/atlas/u/dfh13/bedroom_list.pkl'
        if (os.path.isfile(image_list_filename)):
            self.data_all = pickle.load(open(image_list_filename,'rb'))
        else:
            print "Start finding images"
            start = time.time()
            self.data_all = find_files("/afs/cs.stanford.edu/u/dfh13/LSUN/","*.webp")
            pickle.dump(self.data_all,open(image_list_filename,'wb'))
            print "Finish finding images in", time.time()-start, 's, find',len(self.data_all),'images'

        def train_loop():
            sample_z = np.random.uniform(-1, 1, size=(self.sample_size , self.z_dim))

            self.validation_files = self.data_all[len(self.data_all)-self.validation_size:len(self.data_all)]
            self.sample_files = self.validation_files[:self.batch_size]

            sample_patch1, sample_patch2, sample_dx, sample_dy = get_patches_batch(self.sample_files,0,0,get_patches_2)

            #sample_images = np.array(sample).astype(np.float32)
            data = self.data_all[:len(self.data_all)-self.validation_size]
            counter = 1
            start_time = time.time()
            dx_list = [-1,-1,-1,0,0,1,1,1]
            dy_list = [-1,0,1,-1,1,-1,0,1]
    
            flag = True
    
            for epoch in xrange(25):
                #data = glob(os.path.join("./data", config.dataset, "*.JPEG"))
                batch_idxs = min(len(data), config.train_size)/config.batch_size
    
                for idx in xrange(0, batch_idxs):
                    batch_files = data[idx*config.batch_size:(idx+1)*config.batch_size]
                    #batch = [get_image(batch_file, self.image_size, is_crop=self.is_crop) for batch_file in batch_files]
                    #print "Start loading"
                    #start = time.time()
                    if counter<=500:
                        g_optim = g_optim_l1
                    else:
                        g_optim = g_optim_adv
                    g_optim = g_optim_tot
                    #remainder = (counter // 100) % 8
                    #dx_in = dx_list[remainder]
                    #dy_in = dy_list[remainder]
                    dx_in = 0
                    dy_in = 0
                    patch1_batch, patch2_batch, dx_batch, dy_batch = get_patches_batch(batch_files,dx_in,dy_in,get_patches_2)
    
                    #print 'load time:', time.time()-start
                    #batch_images = np.array(batch).astype(np.float32)
                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
    
                    start = time.time()
                    errD_fake, errD_real, errG_adv, errG_l1, summary_str = self.sess.run([self.d_loss_fake, self.d_loss_real, self.g_loss_adv, self.g_loss_l1, self.sum],
                        feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    self.writer.add_summary(summary_str, counter)
    
                    flag = errG_l1 < 0.1
                    #print 'eval time:', time.time()-start
                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f" \
                        % (epoch, idx, batch_idxs,time.time() - start_time), errD_fake, errD_real, errG_adv, errG_l1)
    
                    
                    errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, g_optim],
                        feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    #self.writer.add_summary(summary_str, counter)
                    
                    if True:
                        errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, d_optim],
                            feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                        #self.writer.add_summary(summary_str, counter)
                        
                        # Update G network
                        '''
                        for i in xrange(8):
                            start = time.time()
                            _, summary_str = self.sess.run([g_optim, self.sum],
                                feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z })
                            self.writer.add_summary(summary_str, counter)
                            #print 'g time:', time.time()-start
                        '''
                        
                        if errD_fake > errG_adv:
                            cnt = 0
                            while errD_fake > errG_adv:
                                errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, d_optim],
                                    feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                                cnt += 1
                                if cnt > 1:
                                    break
                        else:
                            cnt = 0
                            while errG_adv > errD_fake:
                                errD_fake, errG_adv, _ = self.sess.run([self.d_loss_fake, self.g_loss_adv, g_optim],
                                    feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                                cnt += 1
                                if cnt > 1:
                                    break
                        
    
                    # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)
    
                    #errD_fake = self.d_loss_fake.eval({self.patch1: patch1_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    #errD_real = self.d_loss_real.eval({self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
                    #errG = self.g_loss.eval({self.patch1: patch1_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z})
    
                    if np.mod(counter, 30) == 1:
                        print "Generating samples"
                        samples, loss_l1 = self.sess.run(
                            [self.sampler, self.g_loss_l1],
                            feed_dict={ self.patch1: sample_patch1, self.patch2: sample_patch2, self.dx: sample_dx, self.dy: sample_dy, self.z: batch_z}
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
                            feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch, self.dx: dx_batch, self.dy: dy_batch, self.z: batch_z}
                        )
                        save_images(samples, [6, 6],
                                    samples_dir+'/train_%s_%s_synthesized.png' % (epoch, idx))
                        save_images(patch1_batch, [6, 6],
                                    samples_dir+'/train_%s_%s_org.png' % (epoch, idx))
                        save_images(patch2_batch, [6, 6],
                                    samples_dir+'/train_%s_%s_target.png' % (epoch, idx))                    
                        print("[Sample] g_loss_l1: %.8f" % (loss_l1))                    
    
                    #if counter == 10:
                        #self.retrieve()
                        #xxx
                    #if np.mod(counter, 100) == 1:
                        #self.test3(epoch,idx)
                    #if np.mod(counter, 100) == 1:
                        #self.test2(epoch,idx)
                    if np.mod(counter, 3000) == 0:
                        print "Saving checkpoint"
                        self.save(self.checkpoint_dir, counter)
                        print "Save finished"
                    #if counter == 100:
                        #return
            print "Saving checkpoint"
            self.save(config.checkpoint_dir, counter)
            print "Save finished"        
 
        train_loop()
        #self.retrieve()

    def discriminator(self, patch1, patch2, dx, dy, reuse=False, y=None):
        if reuse:
            tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            encoded_patch1 = self.d_encoder2(patch1)
            encoded_patch2 = self.d_encoder(patch2)
            #square_dx = tf.tile(tf.reshape(dx,[-1,1,1,1]),[1,4,4,1])
            #square_dy = tf.tile(tf.reshape(dy,[-1,1,1,1]),[1,4,4,1])
            
            #dx_reshaped = tf.reshape(dx,[-1,1])
            #dy_reshaped = tf.reshape(dy,[-1,1])
            #dxdy = linear(tf.concat(1,[dx_reshaped,dy_reshaped]),self.gf_dim*4*4*4,'d_dxdy')
            #dxdy = tf.reshape(dxdy,[self.batch_size,4,4,-1])
            #print dxdy.get_shape()
            joint = tf.concat(3,[encoded_patch1,encoded_patch2])
            #joint = encoded_patch2
            #print joint.get_shape()

            #h0 = channel_fc(joint, name='d_channel_fc')
            #h0 = lrelu(conv2d(joint, self.df_dim*16, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h0_conv'))
            #h1 = lrelu(self.d_bn1(conv2d(h0, self.df_dim*8, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h1_conv')))
            #h2 = lrelu(self.d_bn2(conv2d(h1, self.df_dim*4, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h2_conv')))
            #h3 = lrelu(self.d_bn3(conv2d(h2, self.df_dim*2, k_h=4, k_w=4, d_h=1, d_w=1, name='d_h3_conv')))
            h3 = joint

            h3_reshape = tf.reshape(h3, [self.batch_size, -1])
            '''
            n_kernels = 100
            dim_per_kernel = 10
            batch_size = self.batch_size
            x = linear(h3_reshape, n_kernels * dim_per_kernel, 'd_h_minibatch')
            activation = tf.reshape(x, (batch_size, n_kernels, dim_per_kernel))

            big = np.zeros((batch_size, batch_size), dtype='float32')
            big += np.eye(batch_size)
            big = tf.expand_dims(big, 1)
        
            abs_dif = tf.reduce_sum(tf.abs(tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)), 2)

            mask = 1. - big
            masked = tf.exp(-abs_dif) * mask

            f = tf.reduce_sum(masked, 2) / tf.reduce_sum(mask)

            #minibatch_features = [f]
            '''

            #dx_reshaped = tf.reshape(dx,[-1,1])
            #dy_reshaped = tf.reshape(dy,[-1,1])
            #h3_reshape = tf.concat(1,[h3_reshape,dx_reshaped,dy_reshaped])

            #h3_1 = tf.nn.relu(linear(h3_reshape,40,'d_h3_1'))
            #h3_2 = tf.nn.relu(linear(h3_1,10,'d_h3_2'))

            h4 = linear(h3_reshape, 1, 'd_h3_lin')

            return tf.nn.sigmoid(h4), h4

    def generator(self, patch1, dx, dy, z, y=None):
        if not self.y_dim:
            # project `z` and reshape
            encoded_patch1 = self.g_encoder(patch1)
            #square_dx = tf.tile(tf.reshape(dx,[-1,1,1,1]),[1,4,4,1])
            #square_dy = tf.tile(tf.reshape(dy,[-1,1,1,1]),[1,4,4,1])
            #dx_reshaped = tf.reshape(dx,[-1,1])
            #dy_reshaped = tf.reshape(dy,[-1,1])
            
            self.z_ = linear(tf.concat(1,[z]), self.gf_dim*8*4*4, 'g_h0_lin')

            self.h0_z = tf.reshape(self.z_, [self.batch_size, 4, 4, self.gf_dim * 8])
            #print encoded_patch1.get_shape(),self.h0_z.get_shape(),square_dx.get_shape()
            h0 = tf.concat(3,[self.h0_z,encoded_patch1])
            #h0 = self.h0_z

            #h0_mixed = channel_fc(h0, name='g_channel_fc')
            #h0_1 = deconv2d(h0, [self.batch_size, 4, 4, self.gf_dim*32], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h0_1')
            #h0_2 = deconv2d(h0_1, [self.batch_size, 4, 4, self.gf_dim*16], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h0_2')
            #h0_3 = deconv2d(h0_2, [self.batch_size, 4, 4, self.gf_dim*8], 4=k_h, k_w=4, d_h=1, d_w=1, name='g_h0_3')
            
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

    def sampler(self, patch1, dx, dy, z, y=None):
        tf.get_variable_scope().reuse_variables()

        if not self.y_dim:
            # project `z` and reshape
            encoded_patch1 = self.g_encoder(patch1)
            #square_dx = tf.tile(tf.reshape(dx,[-1,1,1,1]),[1,4,4,1])
            #square_dy = tf.tile(tf.reshape(dy,[-1,1,1,1]),[1,4,4,1])
            #dx_reshaped = tf.reshape(dx,[-1,1])
            #dy_reshaped = tf.reshape(dy,[-1,1])

            h0_z = tf.reshape(linear(tf.concat(1,[z]), self.gf_dim*8*4*4, 'g_h0_lin'),
                            [-1, 4, 4, self.gf_dim * 8])

            #h0 = tf.concat(3,[encoded_patch1,self.h0_z,square_dx,square_dy])
            h0 = tf.concat(3,[h0_z,encoded_patch1])
            #h0 = h0_z
            #h0_mixed = channel_fc(h0, name='g_channel_fc')
            #h0_1 = deconv2d(h0, [self.batch_size, 4, 4, self.gf_dim*32], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h0_1')
            #h0_2 = deconv2d(h0_1, [self.batch_size, 4, 4, self.gf_dim*16], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h0_2')
            #h0_3 = deconv2d(h0_2, [self.batch_size, 4, 4, self.gf_dim*8], k_h=4, k_w=4, d_h=1, d_w=1, name='g_h0_3')
            
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
            #ckpt_name = 'DCGAN.model-52000'
            #var_list = ['beta1_power_3']
            #init = tf.initialize_variables(var_list=var_list)
            #self.sess.run(tf.init)
            #ckpt_name = 'DCGAN.latest'
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

    def d_encoder(self, image):
        with tf.variable_scope('d_encoder'):
            if self.first_d_encoder:
                self.first_d_encoder = False
            else:
                tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.df_dim,name='d_e_h0_conv'))
            h1 = tf.nn.relu(self.d_e_bn1(conv2d(h0,self.df_dim*2,name='d_e_h1_conv')))
            h2 = tf.nn.relu(self.d_e_bn2(conv2d(h1,self.df_dim*4,name='d_e_h2_conv')))
            h3 = tf.nn.relu(self.d_e_bn3(conv2d(h2,self.df_dim*8,name='d_e_h3_conv')))
    
            return h3

    def d_encoder2(self, image):
        with tf.variable_scope('d_encoder2'):
            '''
            if self.first_encoder:
                self.first_encoder = False
            else:
                tf.get_variable_scope().reuse_variables()
            '''
            h0 = tf.nn.relu(conv2d(image,self.df_dim,name='d_e2_h0_conv'))
            h1 = tf.nn.relu(self.d_e2_bn1(conv2d(h0,self.df_dim*2,name='d_e2_h1_conv')))
            h2 = tf.nn.relu(self.d_e2_bn2(conv2d(h1,self.df_dim*4,name='d_e2_h2_conv')))
            h3 = tf.nn.relu(self.d_e2_bn3(conv2d(h2,self.df_dim*8,name='d_e2_h3_conv')))
    
            return h3

    def encoder_test(self, image):
        with tf.variable_scope('d_encoder'):
            tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.df_dim,name='d_e_h0_conv'))
            h1 = tf.nn.relu(self.d_e_bn1(conv2d(h0,self.df_dim*2,name='d_e_h1_conv'), train=False))
            h2 = tf.nn.relu(self.d_e_bn2(conv2d(h1,self.df_dim*4,name='d_e_h2_conv'), train=False))
            h3 = tf.nn.relu(self.d_e_bn3(conv2d(h2,self.df_dim*8,name='d_e_h3_conv'), train=False))
    
            return h3

    def g_encoder(self, image):
        with tf.variable_scope('g_encoder'):
            if self.first_g_encoder:
                self.first_g_encoder = False
            else:
                tf.get_variable_scope().reuse_variables()
    
            h0 = tf.nn.relu(conv2d(image,self.gf_dim,name='g_e_h0_conv'))
            h1 = tf.nn.relu(self.g_e_bn1(conv2d(h0,self.gf_dim*2,name='g_e_h1_conv')))
            #h1_1 = tf.nn.relu(self.e_bn1_1(conv2d(h1,self.gf_dim*4,d_h=1,d_w=1,name='g_e_h1_1_conv')))
            #h1_2 = tf.nn.relu(self.e_bn1_2(conv2d(h1_1,self.gf_dim*4,d_h=1,d_w=1,name='g_e_h1_2_conv')))
            h2 = tf.nn.relu(self.g_e_bn2(conv2d(h1,self.gf_dim*4,name='g_e_h2_conv')))
            h3 = tf.nn.relu(self.g_e_bn3(conv2d(h2,self.gf_dim*8,name='g_e_h3_conv')))
    
            return h3

    def test1(self,epoch,idx):
        sample_patch1, sample_patch2, sample_dx, sample_dy = get_patches_batch(self.sample_files,0,0,get_patches_2)
        h,w = self.image_shape[:2]
        output = np.zeros([self.batch_size,3*h,3*w,3],dtype='float32')
        output[:,h:2*h,w:2*w,:] = sample_patch1
        dx_list = [-1,-1,-1,0,0,1,1,1]
        dy_list = [-1,0,1,-1,1,-1,0,1]
        for i in xrange(8):
            dx_in = dx_list[i]
            dy_in = dy_list[i]
            sample_dx[:] = dx_in
            sample_dy[:] = dy_in
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
            patch2 = self.sess.run([self.sampler],feed_dict={ self.patch1: sample_patch1, self.dx: sample_dx, self.dy: sample_dy, self.z: batch_z })
            patch2 = np.squeeze(np.array(patch2))
            output[:,(1+dx_in)*h:(2+dx_in)*h,(1+dy_in)*w:(2+dy_in)*w,:] = patch2

        save_images(output, [6, 6],
                eval_dir+'/train_%s_%s.png' % (epoch, idx))

    def test2(self,epoch,idx):
        print 'Running test2'
        sample_patch1, sample_patch2, sample_dx, sample_dy= get_patches_batch(self.sample_files,0,0,get_patches_2)
        h,w = self.image_shape[:2]
        output = np.zeros([self.batch_size,2*h,2*w,3],dtype='float32')
        output[:,h:2*h,w:2*w,:] = sample_patch1
        dx_list = [-0.5,-0.5,0.5,0.5]
        dy_list = [-0.5,0.5,-0.5,0.5]
        for i in xrange(4):
            dx_in = dx_list[i]
            dy_in = dy_list[i]
            sample_dx[:] = dx_in
            sample_dy[:] = dy_in
            batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]) \
                            .astype(np.float32)
            patch2 = self.sess.run([self.sampler],feed_dict={ self.patch1: sample_patch1, self.dx: sample_dx, self.dy: sample_dy, self.z: batch_z})
            patch2 = np.squeeze(np.array(patch2))
            xx = int(dx_in>0)
            yy = int(dy_in>0)
            output[:,xx*h:(1+xx)*h,yy*w:(1+yy)*w,:] = patch2

        save_images(output, [6, 6],
                eval_dir+'/train_%s_%s.png' % (epoch, idx))

    def test3(self,epoch,idx):
        print 'Running test3'
        num_batches = self.validation_size // self.batch_size
        dx_list = [-1,-1,-1,0,0,1,1,1]
        dy_list = [-1,0,1,-1,1,-1,0,1]
        #accuracy_list = []
        dx_batch_ans_list = []
        dy_batch_ans_list = []
        dx_batch_pred_list = []
        dy_batch_pred_list = []

        for i in xrange(num_batches):
            filenames = self.validation_files[i*self.batch_size:(i+1)*self.batch_size]
            patch1_batch, patch2_batch, dx_batch, dy_batch = get_patches_batch(filenames,0,0,get_patches_1)
            dx_batch_ans = dx_batch.copy()
            dy_batch_ans = dy_batch.copy()
            score_all = []
            for j in xrange(8):
                dx_batch[:] = dx_list[j]
                dy_batch[:] = dy_list[j]
                score = self.sess.run([self.D],feed_dict={ self.patch1: patch1_batch, self.patch2: patch2_batch,
                                                        self.dx: dx_batch, self.dy: dy_batch})
                score_all.append(score)
            score_all = np.squeeze(np.array(score_all))
            #print score_all.shape
            label = np.argmax(score_all,axis=0)
            #print label.shape
            dx_batch_pred, dy_batch_pred = label2dxdy(label)

            dx_batch_ans_list.extend(dx_batch_ans)
            dy_batch_ans_list.extend(dy_batch_ans)
            dx_batch_pred_list.extend(dx_batch_pred)
            dy_batch_pred_list.extend(dy_batch_pred)

        #dx_batch_ans_list = np.hstack([dx_batch_ans_list])
        #dy_batch_ans_list = np.hstack([dy_batch_ans_list])
        #dx_batch_pred_list = np.hstack([dx_batch_pred_list])
        #dy_batch_pred_list = np.hstack([dy_batch_pred_list])

        dx_batch_ans_list = np.array(dx_batch_ans_list,dtype='int')
        dy_batch_ans_list = np.array(dy_batch_ans_list,dtype='int')
        dx_batch_pred_list = np.array(dx_batch_pred_list,dtype='int')
        dy_batch_pred_list = np.array(dy_batch_pred_list,dtype='int')
        #accuracy_list = np.array(accuracy_list,dtype='float32')
        accuracy = save_prediction(eval_dir+'/pred_%s_%s.txt' % (epoch, idx),
            dx_batch_ans_list,dy_batch_ans_list,dx_batch_pred_list,dy_batch_pred_list)
        print '[Test3] accuracy:',accuracy

    def retrieve(self):
        from sklearn.neighbors import KDTree
        print 'Running retrieval task'
        psize = 96
        validation_files = self.data_all[len(self.data_all)-self.validation_size:len(self.data_all)]
        sample_files = validation_files[:self.batch_size]
        data = self.data_all[:len(self.data_all)-self.validation_size]

        #sample = [get_image(sample_file, 192, is_crop=True) for sample_file in sample_files]
        #sample_images = np.array(sample).astype(np.float32)
        #save_images(sample_images, [6,6], 'sample_images.png')

        feature = self.encoder_test(self.patch2)
        print feature.get_shape()
        #feature_pooled = tf.reshape(feature,[self.batch_size,-1])
        feature_pooled = feature
        #feature_pooled = tf.nn.max_pool(feature, ksize=[1,4,4,1], strides=[1,1,1,1], padding='VALID')
        print feature_pooled.get_shape()

        feat_list = []
        #img_list = []

        batch_idxs = len(data)/self.batch_size
        for idx in xrange(batch_idxs):
            print idx
            batch_files = data[idx*self.batch_size:(idx+1)*self.batch_size]
            img_batch = [get_image(filename, psize, is_crop=True) for filename in batch_files]
            img_batch = np.array(img_batch).astype(np.float32)
            feat = self.sess.run([feature_pooled],feed_dict={self.patch2:img_batch})
            #feat = np.squeeze(feat)
            feat = np.reshape(feat,[self.batch_size,-1])
            feat_list.extend(feat)
            #img_list.extend(batch_files)

        feat_list = np.array(feat_list)
        #np.save('/atlas/u/dfh13/feat_database',feat_list)
        print 'Dumping result'
        #pickle.dump(feat_list,open('/atlas/u/dfh13/'+self.dataset_name+'_feat_database.pkl','wb'))
        np.save(os.path.join(feat_dir,'feat_database_whole'),feat_list)
        #pickle.dump(img_list,open('/atlas/u/dfh13/'+self.dataset_name+'_img_database.pkl','wb'))
        #print feat_list.shape,feat_list.dtype

        '''
        print 'Building KDTree'
        tree = KDTree(feat_list)
        pickle.dump(tree,open('/atlas/u/dfh13/'+self.dataset_name+'_tree.pkl','wb'))
        
        counter = 0
        val_batch = len(validation_files)/self.batch_size
        for idx in xrange(val_batch):
            batch_files = validation_files[idx*self.batch_size:(idx+1)*self.batch_size]
            img_batch = [get_image(filename, psize, is_crop=True) for filename in batch_files]
            img_batch = np.array(img_batch).astype(np.float32)
            feat = self.sess.run([feature_pooled],feed_dict={self.patch2:img_batch})
            feat = np.squeeze(feat)
            dist, ind = tree.query(feat,k=5)
            
            for i in xrange(6):
                output = []
                for j in xrange(6):
                    row = [img_batch[i*6+j]]
                    for k in ind[i*6+j]:
                        img = get_image(img_list[k], psize, is_crop=True)
                        row.append(img)
                    row_image = np.hstack(row)
                    #print row_image.shape
                    output.append(row_image)
                output_img = np.vstack(output)[None,...]
                #print output_img.shape
                print 'Saving result',counter
                save_images(output_img, [1,1], os.path.join(eval_dir,'result_%d.png'%counter))
                counter += 1
        '''
            
            

