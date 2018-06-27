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
import time
from config_fconv import get_config_fconv
from config_fconv_multi_concat import get_config_fconv_multi_concat,get_concat_iteration_num

def placeholder_inputs(batch_size, num_satelite,num_channels,ntop_candi,model_config=None,num_class=None):
    stars_pl = tf.placeholder(tf.float32,
                              shape=(batch_size, num_satelite, num_channels))
    labels_pl = tf.placeholder(tf.int32,
                                shape=(batch_size, num_satelite))
    sampleweights_pl = tf.placeholder(tf.float32,
                                shape=(batch_size, num_satelite))
    topindices_shuffle_orders = []
    for k in range(1,len(ntop_candi)):
        topindices_shuffle_orders.append( tf.placeholder(tf.int32,shape=(batch_size,ntop_candi[k])) )
    topindices_shuffle_orders = tuple(topindices_shuffle_orders)
    return stars_pl, labels_pl, sampleweights_pl,topindices_shuffle_orders



#def get_model(star_vec,is_training,bn_decay=None,model_config=0,num_class=2):
#    return get_model_fconv(star_vec,is_training,bn_decay,model_config,num_class)
#

def encoder_net(conv_n,net,model_config,num_class,is_training,bn_decay,k_concat,IsMultiConcat,activation_fn=tf.nn.relu):
    batch_size = net.get_shape()[0].value
    num_satelite = net.get_shape()[1].value
    num_elements = net.get_shape()[2].value

    if not IsMultiConcat:
        config_point_encoder,config_group_conv, config_classification = get_config_fconv(model_config,num_elements,num_class,num_satelite)
    else:
        config_point_encoder,config_group_conv, config_classification,concat_ids = get_config_fconv_multi_concat(model_config,num_elements,num_class,num_satelite,k_concat)

    for n in range(len(config_point_encoder)):
        conv_n += 1
        if config_point_encoder[n][4] == 0:
            padding='VALID'
        else:
            padding = 'SAME'

        if n == len(config_group_conv) - 1:
            bn = True
        else:
            bn = True
        net = tf_util.conv2d(net,config_point_encoder[n][2],[config_point_encoder[n][0],config_point_encoder[n][1]],padding=padding,
                            stride=[config_point_encoder[n][3],1],bn=bn,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay,
                             activation_fn=activation_fn)

        if n == concat_ids[k_concat]:
            satelite_point_feat1 = net
        print('point encoder:',net)

    # group feature encoder
    if len(config_group_conv) > 0:
        # by fully convolution
        for n in range(len(config_group_conv)):
            conv_n += 1
            if config_group_conv[n][4] == 0:
                padding='VALID'
            else:
                padding = 'SAME'
            if n==len(config_group_conv)-1:
                bn = True
            else:
                bn = True
            net = tf_util.conv2d(net,config_group_conv[n][2],[config_group_conv[n][0],config_group_conv[n][1]],padding=padding,
                                stride=[config_group_conv[n][3],1],bn=bn,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay,
                                activation_fn=activation_fn)
            print('group feature:',net)
    else:
        # by maxpooling
        net = tf_util.max_pool2d(net,[num_satelite,1],'maxpooling',stride=[1,1],padding='VALID')
        print('max pooling:',net)

    group_feat1 = net

    if model_config == 102:
        group_feat1_expand = tf.tile(group_feat1, [1, num_satelite, 1, 1])
        satelite_fused_feat = tf.concat(axis=3, values=[satelite_point_feat1, group_feat1_expand])
    else:
        group_feat1_expand = tf.tile(tf.reshape(group_feat1, [batch_size, 1, 1, -1]), [1, num_satelite, 1, 1])
        satelite_fused_feat = tf.concat(axis=3, values=[satelite_point_feat1, group_feat1_expand])
    # point feature classification
    # [b,30,1,512]
    net = satelite_fused_feat
    print('satelite_fused_feat: ',satelite_fused_feat)

    return net,satelite_point_feat1,config_classification,conv_n

def get_model_fconv(star_vec,is_training,bn_decay=None,model_config=0,num_class=2,cascade_step=0,IsMultiConcat=False,activation_fn=tf.nn.relu):
    batch_size = star_vec.get_shape()[0].value
    num_satelite = star_vec.get_shape()[1].value
    #num_elements = star_vec.get_shape()[2].value
    print('\n\nget model of cascade step %d'%(cascade_step))
    with tf.variable_scope('cascade_step_'+str(cascade_step)):
        input_data = tf.expand_dims(star_vec, -1)
        # point feature encoder
        conv_n = 0
        net = input_data

        if not IsMultiConcat:
            num_concat_iteration = 1
        else:
            num_concat_iteration = get_concat_iteration_num()[model_config]
        for k_concat in range(num_concat_iteration):
            net,satelite_point_feat1,config_classification,conv_n = encoder_net(conv_n,net,model_config,num_class,is_training,bn_decay,k_concat,IsMultiConcat,activation_fn)

        #net = satelite_feat_concat
        for n in range(len(config_classification)):
            conv_n += 1
            if n == len(config_classification)-1:
                #activation_fn = tf.nn.softmax
                #activation_fn = tf.nn.relu
                activation_fn = None
                #activation_fn = tf.nn.leaky_relu
                #activation_fn = tf.nn.relu
                bn = True
            else:
                activation_fn = activation_fn
                bn = True

            if config_classification[n][4] == 0:
                padding='VALID'
            else:
                padding = 'SAME'

            net = tf_util.conv2d(net,config_classification[n][2],[config_classification[n][0],config_classification[n][1]],padding='VALID',
                                stride=[config_classification[n][3],1],bn=bn,is_training=is_training,scope='conv'+str(conv_n),bn_decay=bn_decay,
                                activation_fn=activation_fn)
            print('classification:',net)
        # [b,30,2]
        net = tf.squeeze(net, [2])
        print('prediction:',net)
    return net

def ntop_slice(last_tensor,top_indices,batch_size,n_top,num_elements):
    batch_idx = tf.reshape(tf.range(batch_size),[batch_size,1])
    batch_idx = tf.expand_dims( tf.tile(batch_idx,[1,n_top]),-1 )
    top_indices = tf.expand_dims( top_indices,-1 )
    top_indices_concat = tf.concat([batch_idx,top_indices],axis=2)
    new_tensor = tf.gather_nd(last_tensor,top_indices_concat)
    return new_tensor

def shuffle_indices(raw_indicies,shuffle_order):
    batch_size = raw_indicies.shape[0]
    sat_num = raw_indicies.shape[1]
    batch_idx = tf.reshape(tf.range(batch_size),[batch_size,1])
    batch_idx = tf.expand_dims( tf.tile(batch_idx,[1,sat_num]),-1 )
    shuffle_order = tf.expand_dims(shuffle_order,-1)
    shuffle_order_concat = tf.concat([batch_idx,shuffle_order],axis=2)
    shuffled_indices = tf.gather_nd(raw_indicies,shuffle_order_concat)
    return shuffled_indices

def set_empty_zero(input,net):
    not_empty = tf.to_float(tf.not_equal(input[:,:,0:1],0))
    net = tf.multiply(net,not_empty)


def get_model_cascade(star_vec,label,sampleweights,is_training_any,ntop_candi,cascade_step,topindices_shuffle_orders,
                      bn_decay=None,model_configs=0,num_class=2,IsMultiConcat=False,activation_fn='relu'):
    '''
    num_pos = 9
    num_satelite  = 56
    ntop_candi = [56,12,11,10,9] : must include  56! to make the code simpler. Can also include 9 or not. Include 9 may be helpful to judge the final conffidency.
    '''
    if activation_fn == 'crelu':
        activation_fn = tf.nn.crelu
    else:
        activation_fn = tf.nn.relu
    batch_size = star_vec.get_shape()[0].value
    num_satelite = star_vec.get_shape()[1].value
    num_elements = star_vec.get_shape()[2].value


    assert ntop_candi[0] == num_satelite
    assert cascade_step < len(ntop_candi)

    is_training_ls = [tf.constant(False)]*(len(ntop_candi))
    is_training_ls[cascade_step] = tf.cond( is_training_any, lambda:True, lambda:False )
    last_star_vec = star_vec+0

    top_indices_ls = []
    cascade_preds = []
    cascade_preds_softmax = []
    cascade_labels = []
    cascade_sampleweights = []
    only_ntop_star_vec_ls = []  # for visulization
    ntop_pred_sorted_ls = []       # for visulization

    max_cascade_num = len(ntop_candi)
    debug_model = {}

    for k,n_top in enumerate(ntop_candi):
        if k == 0:
            new_star_vec = last_star_vec
        else:
            new_star_vec = ntop_slice(last_star_vec,top_indices,batch_size,n_top,num_elements)
            only_ntop_star_vec_ls.append(new_star_vec)
            cascade_labels.append( ntop_slice(cascade_labels[k-1],top_indices,batch_size,n_top,1) )
            cascade_sampleweights.append( ntop_slice(cascade_sampleweights[k-1],top_indices,batch_size,n_top,1) )
            top_indices_ls.append((tf.constant(ntop_candi[k]),top_indices))
        last_star_vec = new_star_vec+0

        pred_k = get_model_fconv(new_star_vec,is_training_ls[k],bn_decay,model_configs[k],num_class,k,IsMultiConcat,activation_fn)

        cascade_preds.append(pred_k)
        pred_sotfmax_k = tf.nn.softmax(pred_k)
        # set all the softmax pred for empty input as zero
        not_empty = tf.to_float(tf.not_equal(new_star_vec[:,:,0:1],0))
        pred_sotfmax_k_remove_empty = tf.multiply(pred_sotfmax_k,not_empty)
        cascade_preds_softmax.append(pred_sotfmax_k_remove_empty)
        #debug_model['test'] = [pred_sotfmax_k,not_empty,pred_sotfmax_k_remove_empty]

        # generate the ntop sliced info () for next cascade step
        if k==0:
            cascade_labels.append( label )
            cascade_sampleweights.append( sampleweights )
        if k==0 or k < max_cascade_num-1:
            top_pred,top_indices_sorted = tf.nn.top_k(pred_sotfmax_k_remove_empty[:,:,1],ntop_candi[k+1],True)

            slice_flag = 'shuffle'
            #slice_flag = 'sat_id_order'
            if slice_flag == 'shuffle':
                top_indices = shuffle_indices(top_indices_sorted,topindices_shuffle_orders[k])
            elif slice_flag == 'score_order':
                top_indices = top_indices_sorted
            elif slice_flag == 'sat_id_order':
                top_indices_minus,_ = tf.nn.top_k(-top_indices_sorted,ntop_candi[k+1],True)
                top_indices = -top_indices_minus

            ntop_pred_sorted_ls.append(top_pred)

    print('\n\ncascade_step = %d'%(cascade_step))
    print('top_indices_ls = ',top_indices_ls)
    print('\n\n')

    debug_model['only_ntop_star_vec_ls'] = only_ntop_star_vec_ls
    debug_model['ntop_pred_sorted_ls'] = ntop_pred_sorted_ls
    debug_model['is_training_ls'] = is_training_ls
    debug_model['cascade_preds_softmax'] = cascade_preds_softmax


    return cascade_preds,cascade_labels,cascade_sampleweights,top_indices_ls,is_training_ls,debug_model

def get_loss_cascade(cascade_preds,cascade_labels,cascade_sampleweights, is_training_ls,ntop_candi,cascade_step):
    loss_ls = []
    loss_active = tf.constant(0.0)
    for k in range( len(ntop_candi) ):
        loss_k = get_loss(cascade_preds[k],cascade_labels[k],cascade_sampleweights[k])
        loss_active += tf.cond(is_training_ls[k],lambda:loss_k,lambda:tf.constant(0.0))
        loss_ls.append(loss_k)
    #loss_sum = tf.reduce_sum( tf.multiply(loss_ls , tf.to_float(is_training_ls) ) )
    return loss_active, loss_ls

def get_loss(pred, label,sampleweights):
    """ pred: B,N,2
        label: B,N """
    loss_classfication = tf.losses.sparse_softmax_cross_entropy(labels=label,logits=pred,weights=sampleweights)
    return loss_classfication

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


def get_pos_idx(label):
    pass

def nargmax(softmax_pred_val,n_top):
    batch_size = softmax_pred_val.shape[0]
    num_sample = softmax_pred_val.shape[1]
    num_class = softmax_pred_val.shape[2]
    assert num_class==2
    argmax_idx = np.zeros((batch_size,n_top))
    for i in range(batch_size):
        sort_idx = np.argsort(-softmax_pred_val[i,:,1])
        argmax_idx[i,:] = sort_idx[0:n_top]
    return argmax_idx

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


    IsVisul = False
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
        pred_scores[i,:,0] = pred_val[i,:,1]
        sort_idx = np.argsort(pred_scores[i,:,0])
        tmp = num_sample-num_pos_label[i]
        pred_logits[i,sort_idx[0:tmp],0] = 0
        pred_logits[i,sort_idx[tmp:],0] = 1
        sorte_idxs[i,:] = sort_idx

    IsVisul = False
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

    return pred_logits[:,:,0]
