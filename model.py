# xyz NOV 2017


import tensorflow as tf
import math
import time
import numpy as np
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'x_utils'))
import tf_util

def placeholder_inputs(batch_size, num_star,num_channels):
    stars_pl = tf.placeholder(tf.float32,
                              shape=(batch_size, num_star, num_channels))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_star))
    return stars_pl, labels_pl


def get_model(star_vec, is_training, bn_decay=None):
    """ ConvNet baseline, input is BxNx4 star direction and flag """
    batch_size = star_vec.get_shape()[0].value
    num_star = star_vec.get_shape()[1].value
    in_channels_num = star_vec.get_shape()[2].value

    # [b,80,4,1]
    input_data = tf.expand_dims(star_vec, -1)
    # CONV

    # [b,80,1,64]
    net = tf_util.conv2d(input_data, 64, [1,in_channels_num], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv1', bn_decay=bn_decay)
   # # [b,30,1,64]
   # net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
   #                      bn=True, is_training=is_training, scope='conv2', bn_decay=bn_decay)
   # # [b,30,1,64]
   # net = tf_util.conv2d(net, 64, [1,1], padding='VALID', stride=[1,1],
   #                      bn=True, is_training=is_training, scope='conv3', bn_decay=bn_decay)
    # [b,30,1,128]
    net = tf_util.conv2d(net, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv4', bn_decay=bn_decay)
    # [b,30,1,1024]
    stars_feat1 = tf_util.conv2d(net, 256, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv5', bn_decay=bn_decay)
    # MAX
    # [b,1,1,1024]
    pc_feat1 = tf_util.max_pool2d(stars_feat1, [num_star,1], padding='VALID', scope='maxpool1')
    # FC
    # [b,1024]
    pc_feat1 = tf.reshape(pc_feat1, [batch_size, -1])
    # [b,256]
    pc_feat1 = tf_util.fully_connected(pc_feat1, 64, bn=True, is_training=is_training, scope='fc1', bn_decay=bn_decay)
    # [b,128]
    pc_feat1 = tf_util.fully_connected(pc_feat1, 32, bn=True, is_training=is_training, scope='fc2', bn_decay=bn_decay)

    # CONCAT
    # [b,30,1,128]
    pc_feat1_expand = tf.tile(tf.reshape(pc_feat1, [batch_size, 1, 1, -1]), [1, num_star, 1, 1])
    # [b,30,1,1152]
    stars_feat1_concat = tf.concat(axis=3, values=[stars_feat1, pc_feat1_expand])

    # CONV
    # [b,30,1,512]
    net = tf_util.conv2d(stars_feat1_concat, 128, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv6')
    # [b,30,1,256]
    net = tf_util.conv2d(net, 32, [1,1], padding='VALID', stride=[1,1],
                         bn=True, is_training=is_training, scope='conv7')
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training, scope='dp1')
    # [b,30,1,2]
    net = tf_util.conv2d(net, 2, [1,1], padding='VALID', stride=[1,1],
                         activation_fn=None, scope='conv8')
    # [b,30,2]
    net = tf.squeeze(net, [2])

    return net


def get_loss(pred, label):
    """ pred: B,N,2
        label: B,N """
    loss_w = {}
    loss_w['classfication'] = tf.constant(1.0)
    loss_w['num_positive'] = tf.constant(0.3)

    # classfication
    loss_classfication_ = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    loss_classfication =  tf.reduce_mean(loss_classfication_)

    #positive num
    num_pos_label = tf.to_float( tf.reduce_sum(label,axis=-1) )
    pred_class = tf.argmax(pred,axis=-1,output_type=tf.int32)
    num_pos_pred = tf.to_float( tf.reduce_sum(pred_class,axis=-1) )
    loss_num_pos = tf.losses.mean_squared_error(num_pos_label,num_pos_pred)
    #loss_num_pos = tf.reduce_mean( tf.squared_difference(num_pos_label,num_pos_pred) )

    loss = loss_classfication * loss_w['classfication'] + loss_num_pos * loss_w['num_positive']
    return loss,loss_classfication,loss_num_pos

if __name__ == "__main__":
    with tf.Graph().as_default():
        a = tf.placeholder(tf.float32, shape=(32,30,9))
        net = get_model(a, tf.constant(True))
        with tf.Session() as sess:
            init = tf.global_variables_initializer()
            sess.run(init)
            start = time.time()
            for i in range(100):
                print(i)
                sess.run(net, feed_dict={a:np.random.rand(32,30,9)})
            print(time.time() - start)
