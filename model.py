import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import cv2


def im2uint8(x):
    if x.__class__ == tf.Tensor:
        return tf.cast(tf.clip_by_value(x, 0.0, 1.0) * 255.0, tf.uint8)
    else:
        t = np.clip(x, 0.0, 1.0) * 255.0
        return t.astype(np.uint8)


class DEBLUR(object):
    def __init__(self, args):
        self.n_levels = 3
        self.scale = 0.5
        self.maxH = args.max_height
        self.maxW = args.max_width
        self.input_path = args.input_path


    def generator(self, inputs, reuse=False, scope='g_net'):
        def ResnetBlock(x, dim, ksize, scope='rb'):
            with tf.variable_scope(scope):
                net = slim.conv2d(x, dim, [ksize, ksize], scope='conv1')
                net = slim.conv2d(net, dim, [ksize, ksize], activation_fn=None, scope='conv2')
                return net

        def DenseBlock(x, dim, ksize, scope='db'):
            with tf.variable_scope(scope):
                net1 = ResnetBlock(x, dim, ksize, scope='d1')
                net2 = ResnetBlock(x+net1, dim, ksize, scope='d2')
                net3 = ResnetBlock(x+net1+net2, dim, ksize, scope='d3')
                net4 = ResnetBlock(x+net1+net2+net3, dim, ksize, scope='d4')
                return x+net1+net2+net3+net4
        n, h, w, c = inputs.get_shape().as_list()

        x_unwrap = []
        
        with tf.variable_scope(scope, reuse=reuse):
            with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                                activation_fn=tf.nn.relu, padding='SAME', normalizer_fn=None,
                                weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True),
                                biases_initializer=tf.constant_initializer(0.0)):

                inp_blur = inputs
                inp_pred = inputs
                for i in range(self.n_levels):
                    scale = self.scale ** (self.n_levels - i - 1)
                    hi = int(round(h * scale))
                    wi = int(round(w * scale))
                    inp_blur = tf.image.resize_images(inputs, [hi, wi], method=0)
                    inp_pred = tf.stop_gradient(tf.image.resize_images(inp_pred, [hi, wi], method=0))
                    inp_all = tf.concat([inp_blur, inp_pred], axis=3, name='inp')

                    # encoder 
                    conv1_1 = slim.conv2d(inp_all, 32, [3, 3], scope='enc1_1_%d' % i)
                    conv1_2 = DenseBlock(conv1_1, 32, 3, scope='enc1_2')
                    conv1_3 = DenseBlock(conv1_2, 32, 3, scope='enc1_2')
                    conv2_1 = slim.conv2d(conv1_3, 64, [3, 3], stride=2, scope='enc2_1_%d' % i)
                    conv2_2 = DenseBlock(conv2_1, 64, 3, scope='enc2_2')
                    conv2_3 = DenseBlock(conv2_2, 64, 3, scope='enc2_2')
                    conv3_1 = slim.conv2d(conv2_3, 128, [3, 3], stride=2, scope='enc3_1_%d' % i)
                    conv3_2 = DenseBlock(conv3_1, 128, 3, scope='enc3_2')
                    conv3_3 = DenseBlock(conv3_2, 128, 3, scope='enc3_2')

                    deconv3_3 = conv3_3

                    # decoder
                    deconv3_2 = DenseBlock(deconv3_3, 128, 3, scope='dec3_2')
                    deconv3_1 = DenseBlock(deconv3_2, 128, 3, scope='dec3_2')
                    deconv2_3 = slim.conv2d_transpose(deconv3_1, 64, [4, 4], stride=2, scope='dec2_3_%d' % i)
                    cat2 = deconv2_3 + conv2_3
                    deconv2_2 = DenseBlock(cat2, 64, 3, scope='dec2_2')
                    deconv2_1 = DenseBlock(deconv2_2, 64, 3, scope='dec2_2')
                    deconv1_3 = slim.conv2d_transpose(deconv2_1, 32, [4, 4], stride=2, scope='dec1_3_%d' % i)
                    cat1 = deconv1_3 + conv1_3
                    deconv1_2 = DenseBlock(cat1, 32, 3, scope='dec1_2')
                    deconv1_1 = DenseBlock(deconv1_2, 32, 3, scope='dec1_2')
                    inp_pred = slim.conv2d(deconv1_1, 1, [3, 3], activation_fn=None, scope='dec1_0_%d' % i)

                    inp_pred = inp_pred + inp_blur

                    if i >= 0:
                        x_unwrap.append(inp_pred)

            return x_unwrap


    def build(self, model_path):
        self.inputs = tf.placeholder(shape=[3, self.maxH, self.maxW, 1], dtype=tf.float32)
        self.outputs = self.generator(self.inputs, reuse=tf.AUTO_REUSE)
        self.sess = tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True)))
        self.saver = tf.train.Saver()
        current_dir = os.path.dirname(os.path.realpath(__file__))
        checkpoint_dir = os.path.join(current_dir, model_path)
        self.saver.restore(self.sess, os.path.join(checkpoint_dir, 'deblur_model'))       


    def test(self):
        input_path = self.input_path
        if os.path.isfile(input_path):
            mode = 'Image'
        else:
            mode = 'Folder'
        if mode == 'Image':
            print(input_path)
            res = self.forward(input_path)
            output_path = input_path[:-4] + '_res' + input_path[-4:]
            cv2.imwrite(output_path, res)
        else:
            imgs = os.listdir(input_path)
            print('Total %d images for deblurring' % len(imgs))
            output_path = input_path + '_res'
            if not os.path.exists(output_path):
                os.makedirs(output_path)
            for i in range(len(imgs)):
                print(imgs[i])
                img_path = os.path.join(input_path, imgs[i])
                res = self.forward(img_path)
                cv2.imwrite(os.path.join(output_path, imgs[i]), res)


    def forward(self, imgpath):
        blur = cv2.imread(imgpath, cv2.IMREAD_UNCHANGED).astype('float32')
        h, w, c = blur.shape
        blur = blur[:,:,::-1]
        if (c == 3):
            blur = blur[:,:,::-1]
        else:
            print('Image is not a color image, return the input image!')
            return blur
        # make sure the width is larger than the height
        rot = False
        if h > w:
            blur = np.transpose(blur,[1,0,2])
            rot = True
            h = blur.shape[0]
            w = blur.shape[1]
        H = self.maxH
        W = self.maxW
        resize = False
        if h > H or w > W:
            scale = min(1.0 * H / h, 1.0 * W / w)
            new_h = int(round(h * scale))
            new_w = int(round(w * scale))
            print('Original Size:', h, w, 'Resize by scale factor', scale, ' to:', new_h, new_w)
            blur = cv2.resize(blur, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            resize = True
            blur_pad = np.pad(blur, ((0, H - new_h), (0, W - new_w), (0, 0)), 'edge')
        else:
            blur_pad = np.pad(blur, ((0, H - h), (0, W - w), (0, 0)), 'edge')
        blur_pad = np.expand_dims(blur_pad, 0)
        blur_pad = np.transpose(blur_pad, (3,1,2,0))
        
        deblur = self.sess.run(self.outputs, feed_dict={self.inputs: blur_pad/255.0})
        res = deblur[-1]
        res = np.transpose(res, (3,1,2,0))
        res = im2uint8(res[0,:,:,:])
        res = res[:,:,::-1]
        # crop the image into original size
        if resize:
            res = res[:new_h,:new_w,:] 
            res = cv2.resize(res, (w, h), interpolation=cv2.INTER_CUBIC);
        else:
            res = res[:h,:w,:]
        if rot:
            res = np.transpose(res,[1,0,2])
        res = res[:,:,::-1]
        return res