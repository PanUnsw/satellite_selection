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

def placeholder_inputs(batch_size, num_satelite,num_channels,model_config=None,num_class=None):
    stars_pl = tf.placeholder(tf.float32,
                              shape=(batch_size, num_satelite, num_channels))
    if model_config!=None and model_config>=200:
        labels_pl = tf.placeholder(tf.int32,
                                    shape=(batch_size, num_class))
        sampleweights_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_class))
    else:
        labels_pl = tf.placeholder(tf.int32,
                                    shape=(batch_size, num_satelite))
        sampleweights_pl = tf.placeholder(tf.float32,
                                    shape=(batch_size, num_satelite))
    return stars_pl, labels_pl, sampleweights_pl


def get_config_fconv(flag,num_elements,num_class):

        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid(0)_or_same(1)
    if flag == 98:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,24,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[56,1,48,56,1]]
        config_classification = [[1,1,24,1,0],
                        [1,1,num_class,1,0]]
    if flag == 99:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,32,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[16,1,64,5,0],
                             [8,1,128,1,0]]
        config_classification = [[1,1,64,1,0],
                                 [1,1,num_class,1,0]]
    if flag == 100:
    #                  [kernel_size0,kernel_size1,num_out_channel]
        config_point_encoder = [[1,num_elements,24,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[11,1,32,3,0],
                             [8,1,48,4,0],
                             [3,1,64,1,0]]
        config_classification = [[1,1,32,1,0],
                                 [1,1,num_class,1,0]]
    if flag == 101:
        config_point_encoder = [[1,num_elements,128,1,0],
                                [1,1,128,1,0]]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[12,1,128,2,0],
                             [11,1,256,2,0],
                             [7,1,512,1,0]]

        config_classification = [[1,1,256,1,0],
                                 [1,1,64,1,0],
                                 [1,1,num_class,1,0]]

    if flag == 102:
        config_point_encoder = [[1, num_elements, 32, 1, 0],
                                [1, 1, 64, 1, 0],
                                ]
        #               kernel_size0,kernel_size1,num_out_channel,stride0,valid_or_same
        config_group_conv = [[17, 1, 64, 1, 0],
                             [15, 1, 128, 2, 0],
                             [13, 1, 256, 1, 0]]

        config_classification = [[1, 1, 128, 1, 0],
                                 [1, 1, 64, 1, 0],
                                 [1, 1, num_class, 1, 0]]

    return config_point_encoder, config_classification, config_group_conv



def get_model(star_vec,is_training,bn_decay=None,model_config=0,num_class=2):
    return get_model_fconv(star_vec,is_training,bn_decay,model_config,num_class)

def get_model_fconv(star_vec,is_training,bn_decay=None,model_config=0,num_class=2):
    batch_size = star_vec.get_shape()[0].value
    num_satelite = star_vec.get_shape()[1].value
    num_elements = star_vec.get_shape()[2].value

    # [b,56,7,1]
    input_data = tf.expand_dims(star_vec, -1)
    # CONV
    conv_n = 0
    config_point_encoder, config_classification,config_group_conv = get_config_fconv(model_config,num_elements,num_class)

    net = input_data
    for n in range(len(config_point_encoder)):
        conv_n += 1
        if config_point_encoder[n][4] == 0:
            padding='VALID'
        else:
            padding = 'SAME'
        net = tf_util.conv2d(net,config_point_encoder[n][2],[config_point_encoder[n][0],config_point_encoder[n][1]],padding=padding,
                             stride=[config_point_encoder[n][3],1],bn=True,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay )

        print('point encoder:',net)
    satelite_point_feat1 = net


    for n in range(len(config_group_conv)):
        conv_n += 1
        if config_group_conv[n][4] == 0:
            padding='VALID'
        else:
            padding = 'SAME'

        net = tf_util.conv2d(net,config_group_conv[n][2],[config_group_conv[n][0],config_group_conv[n][1]],padding=padding,
                            stride=[config_group_conv[n][3],1],bn=True,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay )
        print('group feature:',net)
    group_feat1 = net

    if model_config == 102:
        group_feat1_expand = tf.tile(group_feat1, [1, num_satelite, 1, 1])
        satelite_fused_feat = tf.concat(axis=3, values=[satelite_point_feat1, group_feat1_expand])
    else:
        group_feat1_expand = tf.tile(tf.reshape(group_feat1, [batch_size, 1, 1, -1]), [1, num_satelite, 1, 1])
        satelite_fused_feat = tf.concat(axis=3, values=[satelite_point_feat1, group_feat1_expand])

    # CONV
    # [b,30,1,512]
    net = satelite_fused_feat
    print('satelite_fused_feat: ',satelite_fused_feat)

    #net = satelite_feat_concat
    for n in range(len(config_classification)):
        conv_n += 1
        if n == len(config_classification)-1:
            #activation_fn = tf.nn.softmax
            #activation_fn = tf.nn.relu
            #activation_fn = None
            #activation_fn = tf.nn.leaky_relu
            activation_fn = tf.nn.crelu
        else:
            #activation_fn = tf.nn.relu
            activation_fn = tf.nn.crelu
        if config_classification[n][4] == 0:
            padding='VALID'
        else:
            padding = 'SAME'

        net = tf_util.conv2d(net,config_classification[n][2],[config_classification[n][0],config_classification[n][1]],padding='VALID',
                            stride=[config_classification[n][3],1],bn=True,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay,
                            activation_fn=activation_fn)
        print('classification:',net)
    # [b,30,2]
    net = tf.squeeze(net, [2])
    print('prediction:',net)
    return net


def get_loss(pred, label,sampleweights, loss_w_cs_np=[1.0,0.2],num_class=2):
    """ pred: B,N,2
        label: B,N """
    loss_w = {}
    loss_w['classfication'] = tf.constant(loss_w_cs_np[0])
    loss_w['num_positive'] = tf.constant(loss_w_cs_np[1])

    if num_class > 6:
        loss_classfication = tf.losses.mean_squared_error(labels=label,predictions=pred)
        loss_num_pos = tf.constant(0.0)

    # classfication
    if num_class ==2 or num_class==3:
        loss_classfication = tf.losses.sparse_softmax_cross_entropy(labels=label,logits=pred,weights=sampleweights)

        # positive num loss
        num_pos_label = tf.to_float( tf.reduce_sum(label,axis=-1) )
        pos_num_loss_flag = 'compare'  # 'sum_score', 'true_sum_score' , 'argmax'
        if pos_num_loss_flag == 'compare':
            thres = tf.constant(0.5)
            pred_pos = tf.cast(tf.greater(pred[:,:,1],thres),tf.int32)
            num_pos_pred = tf.reduce_sum(pred_pos,axis=1)
            loss_num_pos = tf.losses.mean_squared_error(num_pos_label,num_pos_pred)

        if pos_num_loss_flag == 'sum_score':
            # positive score sum
            pos_score_sum = tf.reduce_sum(pred[:,:,1],1)
            loss_num_pos = tf.losses.mean_squared_error(pos_score_sum,num_pos_label)

            pos_score_sum_mean = tf.reduce_mean(pos_score_sum)
        if pos_num_loss_flag == 'true_sum_score':
            pos_score_sum = tf.reduce_sum(tf.multiply(pred,label),1)
            neg_score_sum = tf.reduce_sum(tf.multiply(1-pred,1-label),1)
            loss_num_pos = tf.losses.mean_squared_error(pos_score_sum,num_pos_label)
            loss_num_neg = tf.losses.mean_squared_error(neg_score_sum,56-num_pos_label)
            loss_num_pos = (loss_num_pos+loss_num_neg) / 2.0
        if pos_num_loss_flag == 'argmax':
            #positive num by argmax
            pred_class = tf.argmax(pred,axis=-1,output_type=tf.int32)
            num_pos_pred = tf.to_float( tf.reduce_sum(pred_class,axis=-1) )
            loss_num_pos = tf.losses.mean_squared_error(num_pos_label,num_pos_pred)

    if num_class == 1:
        #loss_classfication = tf.losses.mean_squared_error(labels=label,predictions=tf.squeeze(pred,axis=2))
        label1 = tf.expand_dims(label,-1)
        loss_classfication = tf.losses.hinge_loss(labels=label1, logits=pred)
        loss_num_pos = tf.constant(0.0)

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

def get_empty_idx(datas):
    empty_idx = np.zeros(shape=(datas.shape[0],datas.shape[1]))
    for n in range(datas.shape[0]):
        for m in range(datas.shape[1]):
            if np.sum(datas[n,m,0:3]) == 0:
                empty_idx[n,m] = 1
    return empty_idx

def n_argmax(pred_val,label,data):
    '''
    pred_val: [batch_size,num_sample,num_class] num_class=2
    '''
    batch_size = pred_val.shape[0]
    num_sample = pred_val.shape[1]
    num_class = pred_val.shape[2]
    pred_logits = np.zeros(shape=[batch_size,num_sample,1])
    pred_scores = np.zeros(shape=[batch_size,num_sample,1])
    sorte_idxs = np.zeros(shape=[batch_size,num_sample]).astype(np.int)
    num_pos_label = np.sum(label,axis=-1)
    for i in range(pred_val.shape[0]):
        non_zero_idx = np.nonzero(data[i, :, 1])[0]

        num = len(non_zero_idx)

        #pred_scores[i,non_zero_idx,0] = pred_val[i,non_zero_idx,1] - pred_val[i,non_zero_idx,0]
        pred_scores[i, non_zero_idx, 0] = pred_val[i, non_zero_idx, 1]
        sort_idx = np.argsort(pred_scores[i,non_zero_idx,0])
        tmp = num-num_pos_label[i]

        pred_logits[i,non_zero_idx[sort_idx[tmp:]],0] = 1
        sorte_idxs[i,non_zero_idx] = non_zero_idx[sort_idx]


    IsVisul = True
    if IsVisul == True:
        # for visual check
        pred_logits_indipendnet = np.expand_dims(np.argmax(pred_val, 2),axis=-1)
        label_ = np.expand_dims(label,axis=-1)
        fusion_info = np.concatenate([pred_val,pred_logits_indipendnet,pred_scores,pred_logits,label_],axis=-1)
        sorted_fusion_info = fusion_info
        for i in range(batch_size):
            sorted_fusion_info[i,:] = fusion_info[i,sorte_idxs[i,:],:]
        fusion_info_str = ['cls0_score','cls1_score','logi_ind','fus_score','logi_fus']
        #print(np.array2string(sorted_fusion_info[0,:,:],formatter={'float_kind':lambda x:"%6.2f"%x}))
        #import pdb; pdb.set_trace()  # XXX BREAKPOINT

    return pred_logits[:,:,0]


def n_argmax_adv(pred_val,label):
    '''
    pred_val: [batch_size,num_sample,num_class] num_class=2
    '''
    batch_size = pred_val.shape[0]
    num_sample = pred_val.shape[1]
    num_class = pred_val.shape[2]
    pred_logits = np.zeros(shape=[batch_size,num_sample,1])
    pred_scores = np.zeros(shape=[batch_size,num_sample,1])
    sorte_idxs = np.zeros(shape=[batch_size,num_sample]).astype(np.int)
    num_pos_label = np.sum(label,axis=-1)
    for i in range(pred_val.shape[0]):
        pred_scores[i,:,0] = pred_val[i,:,1] - pred_val[i,:,0]
        sort_idx = np.argsort(pred_scores[i,:,0])
        tmp = num_sample-num_pos_label[i]
        pred_logits[i,sort_idx[0:tmp],0] = 0
        pred_logits[i,sort_idx[tmp:],0] = 1
        sorte_idxs[i,:] = sort_idx

    IsVisul = True
    if IsVisul == True:
        # for visual check
        pred_logits_indipendnet = np.expand_dims(np.argmax(pred_val, 2),axis=-1)
        label_ = np.expand_dims(label,axis=-1)
        fusion_info = np.concatenate([pred_val,pred_logits_indipendnet,pred_scores,pred_logits,label_],axis=-1)
        sorted_fusion_info = fusion_info
        for i in range(batch_size):
            sorted_fusion_info[i,:] = fusion_info[i,sorte_idxs[i,:],:]
        fusion_info_str = ['cls0_score','cls1_score','logi_ind','fus_score','logi_fus']
        #print(np.array2string(sorted_fusion_info[0,:,:],formatter={'float_kind':lambda x:"%6.2f"%x}))
        #import pdb; pdb.set_trace()  # XXX BREAKPOINT

    return pred_logits[:,:,0]
