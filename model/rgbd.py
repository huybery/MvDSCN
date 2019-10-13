import tensorflow as tf


class MTV:
    def __init__(self, view_shape, batch_size, ft=False, reg_constant1 = 1.0, reg_constant2 = 1.0, reg = None, \
                denoise = False, model_path = './pretrain/rgbd/ae_fusion', restore_path = './pretrain/rgbd/ae_fusion'):
        self.ft = ft
        self.view1_input = view_shape[0]
        self.view2_input = view_shape[1]
        self.view3_input = view_shape[2]
        self.batch_size = batch_size
        
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0

        # different view feature input
        self.view1 = tf.placeholder(tf.float32, [None, 64, 64, 3], name='view_1')
        self.view2 = tf.placeholder(tf.float32, [None, 64, 64, 1], name='view_2')

        # learning rate
        self.lr = tf.placeholder(tf.float32, [], name='lr')

        # encoder
        # latent is the output of Unet encoder
        latent1 = self.encoder1(self.view1)
        latent2 = self.encoder2(self.view2)

        # lantent_single means output of Dnet encoder
        latent1_single = self.encoder1_single(self.view1)
        latent2_single = self.encoder2_single(self.view2)

        # reshape
        self.z1 = tf.reshape(latent1, [batch_size, -1])
        self.z2 = tf.reshape(latent2, [batch_size, -1])
        z1 = self.z1
        z2 = self.z2

        z1_single = tf.reshape(latent1_single, [batch_size, -1])        
        z2_single = tf.reshape(latent2_single, [batch_size, -1])        

        # self-expressive layer
        # common expressive
        self.Coef = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name = 'Coef')
        # single expressive 
        self.Coef_1 = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name = 'Coef_1')
        self.Coef_2 = tf.Variable(1.0e-8 * tf.ones([self.batch_size, self.batch_size], tf.float32), name = 'Coef_2')
        

        # normalize        
        self.Coef = (self.Coef - tf.diag(tf.diag_part(self.Coef)))

        self.Coef_1 = (self.Coef_1 - tf.diag(tf.diag_part(self.Coef_1)))
        self.Coef_2 = (self.Coef_2 - tf.diag(tf.diag_part(self.Coef_2)))
        
        # zc
        z1_c = tf.matmul(self.Coef, z1)
        z2_c = tf.matmul(self.Coef, z2)

        z1_c_single = tf.matmul(self.Coef_1, z1_single)
        z2_c_single = tf.matmul(self.Coef_2, z2_single)
        
        # reshape
        latent1_c = tf.reshape(z1_c, tf.shape(latent1))
        latent2_c = tf.reshape(z2_c, tf.shape(latent2))

        latent1_c_single = tf.reshape(z1_c_single, tf.shape(latent1_single))
        latent2_c_single = tf.reshape(z2_c_single, tf.shape(latent2_single))

        if self.ft:
            # reconst with self-expressive
            self.view1_r = self.decoder1(latent1_c)
            self.view2_r = self.decoder2(latent2_c)

            self.view1_r_single = self.decoder1_single(latent1_c_single)
            self.view2_r_single = self.decoder2_single(latent2_c_single)
            
        else:
            # only reconst by autoencoder
            self.view1_r = self.decoder1(latent1)
            self.view2_r = self.decoder2(latent2)

            self.view1_r_single = self.decoder1_single(latent1_single)
            self.view2_r_single = self.decoder2_single(latent2_single)
            
        print(latent1.shape, self.view1_r.shape)
        print(latent2.shape, self.view2_r.shape)

        # reconstruction loss  by Unet
        self.reconst_loss_1 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.view1_r, self.view1), 2.0))
        self.reconst_loss_2 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.view2_r, self.view2), 2.0))

        # reconstruction loss by Dnet
        self.reconst_loss_1_single = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.view1_r_single, self.view1), 2.0))
        self.reconst_loss_2_single = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.view2_r_single, self.view2), 2.0))


        self.reconst_loss_single = self.reconst_loss_1_single + self.reconst_loss_2_single
        
        # reconstruction loss all (Unet + Dnet)
        self.reconst_loss = self.reconst_loss_1 + self.reconst_loss_2
        self.reconst_loss += self.reconst_loss_single

        # self-expressive loss by Unet
        self.selfexpress_loss_1 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z1_c, z1), 2.0))
        self.selfexpress_loss_2 = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z2_c, z2), 2.0))
        # self-expressive loss by Dnet
        self.selfexpress_loss_1_single = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z1_c_single, z1_single), 2.0))
        self.selfexpress_loss_2_single = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(z2_c_single, z2_single), 2.0))
        
        # selfexpress all (Unet + Dnet)
        self.selfexpress_loss = self.selfexpress_loss_1 + self.selfexpress_loss_2
        self.selfexpress_loss_single = self.selfexpress_loss_1_single + self.selfexpress_loss_2_single

        self.selfexpress_loss += self.selfexpress_loss_single

        # Coef regularization
        self.reg_loss = tf.reduce_sum(tf.pow(self.Coef, 2.0))

        self.reg_loss += tf.reduce_sum(tf.pow(self.Coef_1, 2.0))
        self.reg_loss += tf.reduce_sum(tf.pow(self.Coef_2, 2.0))


        # unify loss
        self.unify_loss = tf.reduce_sum(tf.abs(tf.subtract(self.Coef, self.Coef_1))) + \
                        tf.reduce_sum(tf.abs(tf.subtract(self.Coef, self.Coef_2))) 


        self.hsic_loss = self.HSIC(self.Coef_1, self.Coef_2)

        # summary loss
        self.loss = self.reconst_loss + reg_constant1 * self.reg_loss + reg_constant2 * self.selfexpress_loss + self.unify_loss * 0.1 + self.hsic_loss * 0.1

        # selfexpression optimizer
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)
        # autoencoder optimizer
        self.optimizer_ae = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.reconst_loss)
        # session
        self.init = tf.global_variables_initializer()
        config  = tf.ConfigProto()  
        config.gpu_options.allow_growth=True
        self.sess = tf.Session(config=config)
        self.sess.run(self.init)
        
        
        self.saver = tf.train.Saver(
            [v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))]
            )

    def HSIC(self, c_v, c_w):
        N = tf.shape(c_v)[0]
        H = tf.ones((N, N)) * tf.cast((1/N), tf.float32) * (-1) + tf.eye(N)
        K_1 = tf.matmul(c_v, tf.transpose(c_v))
        K_2 = tf.matmul(c_w, tf.transpose(c_w))
        rst = tf.matmul(K_1, H)
        rst = tf.matmul(rst, K_2)
        rst = tf.matmul(rst, H)
        rst = tf.trace(rst)
        return rst
        

    def conv_block(self, inputs, out_channels, name='conv'):
        conv = tf.layers.conv2d(inputs, out_channels, kernel_size=3, strides=2, padding="same")
        conv = tf.nn.relu(conv)
        return conv

    def deconv_block(self, inputs, out_channels, name='conv'):
        deconv = tf.layers.conv2d_transpose(inputs, out_channels, kernel_size=3, strides=2, padding='same')
        deconv = tf.nn.relu(deconv)
        return deconv

    def encoder1(self, x):
        net = self.conv_block(x,   64)
        net = self.conv_block(net, 64)
        net = self.conv_block(net, 64)
        return net

    def encoder1_single(self, x):
        net = self.conv_block(x,   64)
        net = self.conv_block(net, 64)
        net = self.conv_block(net, 64)
        return net

    def decoder1(self, z):
        net = self.deconv_block(z,   64)
        net = self.deconv_block(net, 64)
        net = self.deconv_block(net, 3)        
        return net

    def decoder1_single(self, z):
        net = self.deconv_block(z,   64)
        net = self.deconv_block(net, 64)
        net = self.deconv_block(net, 3)        
        return net

    def encoder2(self, x):
        net = self.conv_block(x,   64)
        net = self.conv_block(net, 64)
        net = self.conv_block(net, 64)
        return net

    def encoder2_single(self, x):
        net = self.conv_block(x,   64)
        net = self.conv_block(net, 64)
        net = self.conv_block(net, 64)
        return net

    def decoder2(self, z):
        net = self.deconv_block(z,   64)
        net = self.deconv_block(net, 64)
        net = self.deconv_block(net, 1)        
        return net
    
    def decoder2_single(self, z):
        net = self.deconv_block(z,   64)
        net = self.deconv_block(net, 64)
        net = self.deconv_block(net, 1)        
        return net

    def finetune(self, view1, view2, lr):
        loss, _, Coef, Coef_1, Coef_2 = self.sess.run(
            (self.loss, self.optimizer, self.Coef, self.Coef_1, self.Coef_2), 
            feed_dict={
                self.view1: view1,
                self.view2: view2,
                self.lr: lr
            })
        return loss, Coef, Coef_1, Coef_2

    def reconstruct(self, view1, view2, lr):
        loss, _ = self.sess.run(
            [self.reconst_loss, self.optimizer_ae],
            feed_dict={
                self.view1: view1,
                self.view2: view2,
                self.lr: lr
            }
        )
        return loss
        
    def initlization(self):
        self.sess.run(self.init)

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print("model saved in ", save_path)
        return save_path
    
    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print("mode restored successed.")
    
    def get_latent(self, view1, view2):
        latent_1, latent_2 = self.sess.run(
            [self.z1, self.z2],
            feed_dict={
                self.view1: view1,
                self.view2: view2
            }
        )
        return latent_1, latent_2