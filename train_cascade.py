import time
import pdb, traceback
import tensorflow as tf
import argparse
import math
import h5py
import numpy as np
from reader import Star_Reader
from sklearn.preprocessing import OneHotEncoder
from model_cascade import get_model_cascade,get_loss_cascade,placeholder_inputs,n_argmax_adv
import time
#import matplotlib.pyplot as plt
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
T_START = time.time()


parser = argparse.ArgumentParser()

parser.add_argument('--a', default='cascade', help='the name of configuration')

parser.add_argument('--feed_star_elements', default='xyzhd', help='part of xyzhdgn')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 24]')
parser.add_argument('--neg_loss_w',default='0.85-0.35',help='loss weight for negative sample')
parser.add_argument('--empty_loss_w',default=0.05,help='loss weight for empty sample')
parser.add_argument('--loss_w_class', type=float, default=1.0, help='The loss weight for classification')
parser.add_argument('--loss_w_numpos', type=float, default=0.0, help='The loss weight for positive number constrain')
parser.add_argument('--num_pos_ls',default='all',help='7-9-11-13 positive num list of data to be selected')
parser.add_argument('--model_config',type=str,default='1-1',help='model config: 101, 6, 7')
parser.add_argument('--activation_fn',default='relu',help='activation_fn')
parser.add_argument('--UseMultiConcat',action='store_true',help='use multiple concat')
parser.add_argument('--data_source',default='data_WGDOP_small',help='data_sync or data_withg')
parser.add_argument('--data_source_test',default='data_WGDOP_test',help='data_sync or data_withg')

parser.add_argument('--UseErrCondLabel',action='store_true',help='UseErrCondLabel')
parser.add_argument('--UseEmptyLabel',action='store_true',help='Use empty as a seperate label')
parser.add_argument('--IsRegression',action='store_true',help='Use positve satelite index as label')
parser.add_argument('--IsHingeloss',action='store_true',help='Use positve satelite index as label')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=3, help='Epoch to run [default: 50]')
parser.add_argument('--learning_rate', type=float, default=0.1, help='Initial learning rate [default: 0.096]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=500000, help='Decay step for lr decay [default: 400000]')
parser.add_argument('--decay_rate', type=float, default=0.75, help='Decay rate for lr decay [default: 0.73]')
parser.add_argument('--bn_decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')



parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--finetune',action='store_true',help='finetune')
parser.add_argument('--fix_pn',action='store_true',help='constrain the positive number forcely')
parser.add_argument('--IsShuffleSats',action='store_true',help='IsShuffleSats')
parser.add_argument('--model_epoch', default='2-2', help='The epoch of model to use')
parser.add_argument('--plotsaved',action='store_true',help='only plotsaved')

parser.add_argument('--ntop_candi',default='12',help='cascade approach: 12-11-10-9')
parser.add_argument('--cascade_step', type=int, default=0, help='0,1,2')

FLAGS = parser.parse_args()
FLAGS.IsShuffleSats = True
ISDEBUG = False

CASCADE_STEP = FLAGS.cascade_step
BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate


if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_eval.txt'
    LOG_DIR = FLAGS.log_dir
elif FLAGS.finetune:
    log_name = 'log_train_cascade_%d.txt'%(CASCADE_STEP)
    LOG_DIR = FLAGS.log_dir
else:
    log_name = 'log_train_cascade_%d.txt'%(CASCADE_STEP)
    if 'LOG' in FLAGS.log_dir:
        LOG_DIR = FLAGS.log_dir
    else:
        LOG_DIR = FLAGS.log_dir+FLAGS.a+'_'+FLAGS.feed_star_elements+'_b'+str(BATCH_SIZE)+'_'+\
            FLAGS.num_pos_ls+'_'+FLAGS.data_source+'_ntop'+FLAGS.ntop_candi+'_mc'+str(FLAGS.model_config) #+'_'+FLAGS.activation_fn
        if FLAGS.UseMultiConcat:
            LOG_DIR += 'MtiCon'
    if FLAGS.IsHingeloss:
        LOG_DIR += '_HLoss'
LOG_DIR = os.path.join(BASE_DIR+'/RES',LOG_DIR)

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not FLAGS.only_evaluate:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
else:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'a')

FUSION_LOG_FOUT = open(os.path.join(BASE_DIR+'/RES', 'fusion_log.txt'), 'a')


FLAGS.model_epoch = [int(s) for s in  FLAGS.model_epoch.split('-')]
FLAGS.model_config = [int(s) for s in  FLAGS.model_config.split('-')]
FLAGS.neg_loss_w = [float(s) for s in  FLAGS.neg_loss_w.split('-')]

if FLAGS.ntop_candi=='':
    FLAGS.ntop_candi = [56]
else:
    FLAGS.ntop_candi = [56]+[int(s) for s in FLAGS.ntop_candi.split('-')]
if FLAGS.num_pos_ls == 'all':
    FLAGS.num_pos_ls = None
else:
    FLAGS.num_pos_ls = FLAGS.num_pos_ls.split('-')
    FLAGS.num_pos_ls = [int(s) for s in FLAGS.num_pos_ls]
if FLAGS.UseErrCondLabel:
    NUM_CLASSES = 3
    print('**** Using Error condition label  num_class=3')
elif FLAGS.UseEmptyLabel:
    NUM_CLASSES = 3
    print('**** Using empty as a new label  num_class=3')
elif FLAGS.IsRegression:
    NUM_CLASSES = 9
elif FLAGS.IsHingeloss:
    NUM_CLASSES = 2
else:
    NUM_CLASSES = 2
POSITIVE_LABEL = 1

BN_INIT_DECAY = 0.5
#BN_DECAY_DECAY_RATE = 0.5

BN_DECAY_DECAY_RATE = FLAGS.bn_decay_rate
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.999999



# Load data
if FLAGS.only_evaluate:
    training_data_rate = 0.0
else:
    training_data_rate = 0.95
IsSaveRes = True
NUM_STAR = Star_Reader.max_num_instars
star_reader_train = Star_Reader(train_rate=1,data_source=FLAGS.data_source,
                          IsOnlyEval=FLAGS.only_evaluate,IsUseErrCondLabel=FLAGS.UseErrCondLabel,
                          IsEmptyLabel=FLAGS.UseEmptyLabel,IsRegression=FLAGS.IsRegression)
star_reader_test = Star_Reader(train_rate=0,data_source=FLAGS.data_source_test,
                          IsOnlyEval=FLAGS.only_evaluate,IsUseErrCondLabel=FLAGS.UseErrCondLabel,
                          IsEmptyLabel=FLAGS.UseEmptyLabel,IsRegression=FLAGS.IsRegression)
NUM_CHANNEL = len(FLAGS.feed_star_elements)
train_data,train_label,_,_ = star_reader_train.get_train_test(
    feed_star_elements=FLAGS.feed_star_elements,num_pos_ls=FLAGS.num_pos_ls)
_,_, test_data,test_label = star_reader_test.get_train_test(
    feed_star_elements=FLAGS.feed_star_elements,num_pos_ls=FLAGS.num_pos_ls)





T_LOADDATA = time.time()
print('load data t: %d ms'%(1000*(T_LOADDATA-T_START)))

ele_histogram,bins = np.histogram( train_data[:,:,0],bins='auto' )
#plt.plot(bins[1:],ele_histogram[1:])
#plt.show()

#print(np.shape(train_data))
#index = 1
#train_data_new = np.reshape(train_data, (-1, 2))
#train_tmp = np.nonzero(train_data_new[:,index])
#train_tmp = np.transpose(train_tmp)

#print(np.shape(train_tmp))
#plt.hist(train_data_new[train_tmp,index], bins='auto')  # arguments are passed to np.histogram
#plt.title("Histogram with 'auto' bins")
#plt.show()


fo_missed_intop = open(os.path.join(LOG_DIR,'missed_intop.txt'),'w')
all_eles =  ['t']+[e for  e in  FLAGS.feed_star_elements]
all_eles += ['label','pre_logi','pre_v','pred-softmax']
head_str = ''.join(['%10s'%s for s in all_eles])
fo_missed_intop.write(head_str+'\n\n')
fo_missed_intop.flush()

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string(str(FLAGS)+'\n')
log_string(star_reader_train.data_summary_str)
log_string(star_reader_test.data_summary_str)

def get_learning_rate(batch):
    learning_rate = tf.train.exponential_decay(
                        BASE_LEARNING_RATE,  # Base learning rate.
                        batch * BATCH_SIZE,  # Current index into the dataset.
                        DECAY_STEP,          # Decay step.
                        DECAY_RATE,          # Decay rate.
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!!
    return learning_rate

def get_bn_decay(batch):
    bn_momentum = tf.train.exponential_decay(
                      BN_INIT_DECAY,
                      batch*BATCH_SIZE,
                      BN_DECAY_DECAY_STEP,
                      BN_DECAY_DECAY_RATE,
                      staircase=True)
    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def train():
    t0_train = time.time()
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            satelites_pl, labels_pl, sampleweights_pl,topindices_shuffle_orders = placeholder_inputs(BATCH_SIZE, NUM_STAR, NUM_CHANNEL,FLAGS.ntop_candi,model_config=FLAGS.model_config,num_class=NUM_CLASSES)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0,name='global_step',trainable=False)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if FLAGS.IsHingeloss:
                model_num_class = NUM_CLASSES - 1
            else:
                model_num_class = NUM_CLASSES
            cascade_preds,cascade_labels,cascade_sampleweights,top_indices_ls,is_training_ls,debug_model = get_model_cascade(
                    satelites_pl,labels_pl,sampleweights_pl,is_training_pl,FLAGS.ntop_candi,
                    CASCADE_STEP, topindices_shuffle_orders, bn_decay=bn_decay,model_configs=FLAGS.model_config,
                    num_class=model_num_class,IsMultiConcat=FLAGS.UseMultiConcat,activation_fn=FLAGS.activation_fn )
            t1_train = time.time()
            print('get_model_cascade t: %d ms'%(1000*(t1_train-t0_train)))
            loss, loss_ls = get_loss_cascade(cascade_preds, cascade_labels,cascade_sampleweights,is_training_ls,FLAGS.ntop_candi,CASCADE_STEP)
            t2_train = time.time()
            print('get_loss_cascade t: %d ms'%(1000*(t2_train-t1_train)))

            active_pred = cascade_preds[CASCADE_STEP]
            active_label = cascade_labels[CASCADE_STEP]
            tf.summary.scalar('loss', loss)
            #tf.summary.histogram('pred_hist',pred)

            if FLAGS.IsRegression:
                correct = tf.equal(tf.to_int64(active_pred), tf.to_int64(active_label))
            else:
                correct = tf.equal(tf.argmax(active_pred, 2), tf.to_int64(active_label))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*FLAGS.ntop_candi[CASCADE_STEP])
            tf.summary.scalar('accuracy', accuracy)

            # Get training operator
            learning_rate = get_learning_rate(batch)
            tf.summary.scalar('learning_rate', learning_rate)
            if OPTIMIZER == 'momentum':
                optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=MOMENTUM)
            elif OPTIMIZER == 'adam':
                optimizer = tf.train.AdamOptimizer(learning_rate)
            train_op = optimizer.minimize(loss, global_step=batch)

            # Add ops to save and restore all the variables.
            savers = []
            for cascade_step in range(len(FLAGS.ntop_candi)):
                var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='cascade_step_'+str(cascade_step))
                var_list += [batch]
                saver_k = tf.train.Saver(var_list=var_list,max_to_keep=50)
                savers.append(saver_k)

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        # Add summary writers
        merged = tf.summary.merge_all()
        if not FLAGS.only_evaluate:
            train_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'train'),
                                    sess.graph)
            test_writer = tf.summary.FileWriter(os.path.join(LOG_DIR, 'test'))
        else:
            test_writer = None

        # Init variables
        init = tf.global_variables_initializer()
        sess.run(init, {is_training_pl:True})

        ops = {'satelites_pl': satelites_pl,
               'labels_pl': labels_pl,
               'sampleweights_pl':sampleweights_pl,
               'is_training_pl': is_training_pl,
               'active_pred': active_pred,
               'active_label': active_label,
               'cascade_preds': cascade_preds,
               'cascade_labels': cascade_labels,
               'top_indices_ls':top_indices_ls,
               'loss': loss,
               'loss_ls': loss_ls,
               'loss_num_pos':loss,
               'loss_classfication':loss,
               'train_op': train_op,
               'merged': merged,
               'step': batch,
               'topindices_shuffle_orders':topindices_shuffle_orders,
               'debug_model': debug_model}

        print('load train graph t: %d ms'%(1000*(time.time()-t0_train)))


        for cascade_step in range(CASCADE_STEP+1):
            if FLAGS.finetune or FLAGS.only_evaluate:
                MODEL_PATH = os.path.join(LOG_DIR,'model_cascade-%d_config-%d.ckpt-%d'%(cascade_step,FLAGS.model_config[cascade_step],FLAGS.model_epoch[cascade_step]))
                savers[cascade_step].restore(sess,MODEL_PATH)
                log_string('\n*******************************')
                if FLAGS.finetune:
                    log_string('fine tune, restore model from: \n\t%s\n'%(MODEL_PATH))
                if FLAGS.only_evaluate:
                    log_string('only evaluate, restore model from: \n\t%s'%(MODEL_PATH))
                log_string('*******************************')
            else:
                if cascade_step < CASCADE_STEP:
                    MODEL_PATH = os.path.join(LOG_DIR,'model_cascade-%d_config-%d.ckpt-%d'%(cascade_step,FLAGS.model_config[cascade_step],FLAGS.model_epoch[cascade_step]))
                    savers[cascade_step].restore(sess,MODEL_PATH)
                    sess.run( batch.assign(0) )
                    log_string('\n*******************************')
                    log_string('cascade_step=%d, restore model from: \n\t%s'%(cascade_step,MODEL_PATH))
                    log_string('*******************************')
            log_string('cascade_step = %d, global_sep=%d, learning_rate=%f, bn_decay=%f\n' % ( cascade_step,sess.run(batch), sess.run(learning_rate), sess.run(bn_decay)))

        start_epoch = 0
        if FLAGS.finetune:
            start_epoch += FLAGS.model_epoch[CASCADE_STEP]+1
        for epoch in range(start_epoch,start_epoch+MAX_EPOCH):
            log_string('\n**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()


            if not FLAGS.only_evaluate:
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch)
            else:
                train_log_str=''


            eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch)

            # Save the variables to disk.
            if ( epoch >= 0 and (epoch) % 2 == 0 ) and (not FLAGS.only_evaluate):
                save_path = savers[CASCADE_STEP].save(sess, os.path.join(LOG_DIR, "model_cascade-%d_config-%d.ckpt"%(CASCADE_STEP,FLAGS.model_config[cascade_step])),global_step=epoch)
                log_string('\n*******************************')
                log_string("Model saved in file: %s" % (save_path) )
                log_string('CASCADE_STEP = %d, global_sep=%d, learning_rate=%f, bn_decay=%f\n'%(CASCADE_STEP,sess.run(batch),sess.run(learning_rate),sess.run(bn_decay)))
                log_string('*******************************')
            if epoch == MAX_EPOCH-1 and not FLAGS.only_evaluate:
                FUSION_LOG_FOUT.write(str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n')


TimeStep_ERR_HIST_N = 8
TP_ERR_HIST_N = 4

def check_data0_label1(cur_data,cur_label,flag=''):
    err_n = 0
    for b in range(cur_data.shape[0]):
        for s in range(cur_data.shape[1]):
            if cur_data[b,s,0] == 0 and cur_label[b,s]==1:
                err_n += 1
    if err_n == 0:
        print(flag+'  no 0 data with 1 label')
    else:
        print(flag+' err %d sats  data with 1 label'%(err_n))

def set_zero_all_neg(pred_logits):
    pred_neg = pred_logits != POSITIVE_LABEL
    for i in range(pred_logits.shape[0]):
        pred_logits[i,pred_neg[i,:]] = 0
    return pred_logits

def get_TPFN(pred_logits,label,label_miss_num=0):
        pred_logits = set_zero_all_neg(pred_logits)
        Pred_True = (pred_logits == label)
        Pred_Pos =  (pred_logits == POSITIVE_LABEL)
        TP = Pred_Pos * Pred_True
        TN = Pred_True * (1 - Pred_Pos)
        FN = (1 - Pred_True) * (1 - Pred_Pos)
        FP = (1 - Pred_True) * Pred_Pos
        num_TP_TN_FN_FP = np.array( [ np.sum(TP), np.sum(TN),np.sum(FN)+label_miss_num, np.sum(FP)])
        pos_num = np.sum(Pred_Pos==POSITIVE_LABEL,axis=1)
        #correct = np.sum(Pred_True)

        # the err time steps number histogram
        pred_errnum = np.sum(Pred_True==False,axis=1) + label_miss_num
        times_errn_histogram = np.histogram(pred_errnum,bins=range(TimeStep_ERR_HIST_N+1))[0]

        # the err satelite number histogram
        TP_err_histogram = np.histogram( np.sum(TP,axis=1)-np.sum(label==POSITIVE_LABEL,axis=1)+label_miss_num,
                                        bins=range(-TP_ERR_HIST_N,2) )[0]
        return num_TP_TN_FN_FP, pos_num, times_errn_histogram,TP_err_histogram

def cal_accu(num_TP_TN_FN_FP):
    num_TP_TN_FN_FP = num_TP_TN_FN_FP.astype(np.float32)
    P = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[3]
    real_P = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[2]
    recall = num_TP_TN_FN_FP[0] / real_P
    precision = num_TP_TN_FN_FP[0] / P
    T = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[1]
    correct = T / np.sum(num_TP_TN_FN_FP)
    return recall, precision,correct

def get_ntop_pos_num_err_str(top_indices,active_label):
    ntop = top_indices[0]
    err_pos_num_intop = []
    for k in range(active_label.shape[0]):
        pos_num_intop = np.sum(active_label[k,top_indices[1][k,:]]==POSITIVE_LABEL)
        pos_num_real = np.sum(active_label[k,:]==POSITIVE_LABEL)
        err_pos_num_intop.append( np.absolute(pos_num_real-pos_num_intop) )
    err_pos_num_intop = np.array(err_pos_num_intop)
    mean_err_pos_num_intop = np.mean(err_pos_num_intop)
    #mean_err_pos_num_intop_str = '\nmean pos num err in %d top: %0.7f'%(ntop,mean_err_pos_num_intop)

    err_pos_num_intop_idx = np.nonzero( err_pos_num_intop )[0]
    return mean_err_pos_num_intop,ntop,err_pos_num_intop_idx

def dolog(tot,epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,
          t_per_timestep,times_errn_histogram,TP_err_histogram,mean_err_pos_num_intop,ntop,cascade_step):
    if FLAGS.IsRegression:
        loss_str = np.array_str(loss_sum[0],precision=3)
        log_string(loss_str)
        return loss_str

    if cascade_step==0:
        logstr = '\n--------------------------------------------------------------\n'
    else:
        logstr = '\n'
    logstr += 'cascade_step: %d'%(cascade_step)
    if CASCADE_STEP == cascade_step:
        logstr += '\tactive\n'

    elif cascade_step == len(FLAGS.ntop_candi):
        logstr += '\toverall\n'
        #return logstr
    else:
        logstr += '\n'
        #return logstr
    recall, precision,correct = cal_accu(num_TP_TN_FN_FP)
    ave_num_pos_err =  num_pos_err_sum/float(BATCH_SIZE)
    loss_str = np.array_str(loss_sum,precision=3)
    logstr += '%s [%d-%d] t:%0.2g ms \tloss[ave,cls,num]: %s \tnum_err: %0.3g' % (
        tot,epoch,batch_idx,t_per_timestep*1000,loss_str,ave_num_pos_err)
    logstr += '\treca: %0.3g, prec: %0.3g, accu: %0.4f'% (
        recall,precision,correct)

    times_errn_histogram_rate = times_errn_histogram / np.sum(times_errn_histogram)
    errn_his_str = np.array2string(times_errn_histogram,formatter={'int':lambda x: "%5.4g"%x})
    errn_his_rate_str = np.array2string(times_errn_histogram_rate,formatter={'float_kind':lambda x:"%.2g"%x})
    TP_err_histogram_rate = TP_err_histogram.astype(np.float32) / np.sum(TP_err_histogram)
    TP_err_histogram_rate_str = np.array2string(TP_err_histogram_rate,formatter={'float_kind':lambda x:"%.2g"%x})
    logstr += '\ntimes err: %s \t TP err: %s'%(errn_his_rate_str,TP_err_histogram_rate_str)
    if ntop!=None and mean_err_pos_num_intop!=None:
        mean_err_pos_num_intop_str = '\nmean pos num err in %d top: %0.7f'%(ntop,mean_err_pos_num_intop)
        logstr += mean_err_pos_num_intop_str

    log_string(logstr)
    return logstr

def get_sample_loss_weights(label,data):
    slw = np.ones(shape=label.shape)
    for i in range(label.shape[0]):
        for j in range(label.shape[1]):
            if label[i,j] == 0:
                if data[i,j,0] == 0:
                    slw[i,j] = FLAGS.empty_loss_w
                else:
                    slw[i,j] = FLAGS.neg_loss_w[CASCADE_STEP]
    return slw

def shuffle_train(train_data,train_label):
    N = train_data.shape[0]
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    train_data = train_data[idxs,:,:]
    train_label = train_label[idxs,:]
    return train_data,train_label

def predval_2_predlogit(pred_val,label=None):
    if  FLAGS.IsRegression:
        pred_logit = pred_val.astype(np.int)
    elif FLAGS.IsHingeloss:
        pred_logit = np.squeeze((pred_val>0.5).astype(np.int),-1)
    elif FLAGS.fix_pn:
        pred_logit = n_argmax_adv(pred_val,label)
    else:
        pred_logit = np.argmax(pred_val, 2)
    return pred_logit

def shuffle_sats(data,label):
    sat_num = data.shape[1]
    for i in range(data.shape[0]):
        shuffled_sat_idx = np.arange(sat_num)
        np.random.shuffle(shuffled_sat_idx)
        data[i,:,:] = data[i,shuffled_sat_idx,:]
        label[i,:] = label[i,shuffled_sat_idx]
    return data,label

def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_STAR,:], train_label[:,0:NUM_STAR])
    current_data = train_data
    current_label = train_label
    current_data,current_label = shuffle_train(current_data,current_label)

    num_batches = current_data.shape[0] // BATCH_SIZE

    #print('total batch num = ',num_batches)
    t_per_timestep = -1
    for batch_idx in range(num_batches):
        t0 = time.time()
        #if batch_idx % 100 == 0:
            #print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        data_i = current_data[start_idx:end_idx, :, :]
        label_i = current_label[start_idx:end_idx]
        data_i,label_i = shuffle_sats(data_i,label_i)

        sample_loss_weight_i = get_sample_loss_weights(label_i,data_i)

        feed_dict = {ops['satelites_pl']: data_i,
                     ops['labels_pl']: label_i,
                     ops['sampleweights_pl']:sample_loss_weight_i,
                     ops['is_training_pl']: is_training,
                     ops['topindices_shuffle_orders']:gen_topindices_shuffle_orders(BATCH_SIZE) }

        summary, step, _, loss_val, cascade_preds,cascade_preds_softmax,cascade_labels,loss_classfication,loss_num_pos,\
            top_indices_ls,active_label,ntop_pred_sorted_ls = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['cascade_preds'],ops['debug_model']['cascade_preds_softmax'],ops['cascade_labels'],
                        ops['loss_classfication'],ops['loss_num_pos'],ops['top_indices_ls'],ops['active_label'],ops['debug_model']['ntop_pred_sorted_ls']],
                     feed_dict=feed_dict)
        train_writer.add_summary(summary, step)

        if not FLAGS.fix_pn:
            cascade_pred_logits = [predval_2_predlogit(pred_val) for pred_val in cascade_preds]
        else:
            cascade_pred_logits = []
            for i in range(len(cascade_preds_softmax)):
                cascade_pred_logits.append( predval_2_predlogit(cascade_preds_softmax[i],cascade_labels[i]) )

        if ISDEBUG:
            only_ntop_star_vec_ls,is_training_ls = sess.run(
                [ops['debug_model']['only_ntop_star_vec_ls'],
                 ops['debug_model']['is_training_ls']],
                                             feed_dict=feed_dict)
        #*********************** evaluation metrics

        if (epoch == 0 and batch_idx <= 10) or (batch_idx>0 and batch_idx%10==0):
            for m in range(len(cascade_labels)+1):
                #if m != CASCADE_STEP and m!=len(cascade_labels):
                if m != CASCADE_STEP:
                    continue
                total_correct = 0
                total_seen = 0
                loss_sum = np.zeros(shape=(3),dtype=np.float)

                num_TP_TN_FN_FP = np.array([0,0,0,0])
                times_errn_histogram = np.zeros(shape=TimeStep_ERR_HIST_N)
                TP_err_histogram = np.zeros(shape=TP_ERR_HIST_N+1)
                num_pos_err_sum = 0

                if m == len(cascade_labels):
                    k = m-1
                    #label_miss_num = np.sum(label_i) - np.sum(cascade_labels[k])
                else:
                    k = m
                    label_miss_num = 0
                pred_logit = cascade_pred_logits[k]
                active_label = cascade_labels[k]

                num_TP_TN_FN_FP_i,num_pos_pred,times_errn_histogram_i,TP_err_histogram_i = get_TPFN(
                            pred_logit,active_label,label_miss_num)
                num_TP_TN_FN_FP = num_TP_TN_FN_FP_i
                times_errn_histogram += times_errn_histogram_i
                TP_err_histogram += TP_err_histogram_i

                num_pos_label = np.sum(active_label==POSITIVE_LABEL,axis=-1)
                num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
                num_pos_err_sum += num_pos_err

                if k < len(cascade_labels)-1:
                    mean_err_pos_num_intop,ntop_k,err_pos_num_intop_idx = get_ntop_pos_num_err_str(top_indices_ls[k],active_label)
                    if k == 0 and epoch>=20 and mean_err_pos_num_intop>0.00 and mean_err_pos_num_intop<=0.06:
                        Write_MissedIntop(fo_missed_intop,epoch,batch_idx,data_i,active_label,cascade_preds[k],cascade_preds_softmax[k],
                                          pred_logit,top_indices_ls[k],err_pos_num_intop_idx,mean_err_pos_num_intop )

                else:
                    mean_err_pos_num_intop=None
                    ntop_k = None

                total_seen += (BATCH_SIZE*NUM_STAR)
                loss_sum += np.array([loss_val,loss_classfication,loss_num_pos])
                t_per_timestep = (time.time()-t0)/BATCH_SIZE

                dolog('train',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,t_per_timestep,
                    times_errn_histogram,TP_err_histogram,mean_err_pos_num_intop,ntop_k,m)

                #if k==0 and err_pos_num_intop_idx.shape[0]>0:
                    #import pdb; pdb.set_trace()
                    #print('\n\nerr_pos_num_intop_idx= %d\n\n'(err_pos_num_intop_idx.shape[0]))
                if  False and (FLAGS.IsRegression or FLAGS.IsHingeloss or True):
                    if m==0 or m<CASCADE_STEP:
                        log_string('ntop pred  : %s'%(np.array_str(ntop_pred_sorted_ls[k][0,0:16],precision=2)))
                    else:
                        log_string('raw pred  : %s'%(np.array_str(cascade_preds[k][0,0:16,1],precision=2)))
                    #print('label : %s\n'%(np.array_str(label_i[0,0:10],precision=2)))


        #******************************** debug info
                if ISDEBUG and  k==0 and mean_err_pos_num_intop>0:
                    vis_time_step = err_pos_num_intop_idx[0]
                    vis_sat_n = -1
                    last_data_i = data_i[0,:,:]
                    #print('is_training_ls: ',is_training_ls)
                    print('\n***\ncascade %d, ntop=%d'%(k,FLAGS.ntop_candi[k]))
                    print('next ntop: %d'%(FLAGS.ntop_candi[k+1]))
                    top_indices = top_indices_ls[k][1]
                    print('ntop indicates: ',top_indices[0,:])
                    print('ntop pred: ',ntop_pred_sorted_ls[k][0,:])

                    new_star_vec = only_ntop_star_vec_ls[k]
                    new_data_i = last_data_i[top_indices[0,:],:]
                    last_data_i = new_data_i
                    try:
                        assert (new_star_vec[0,:,:] == new_data_i).all()
                        print('new_star_vec right, k=%d'%(k))
                    except:
                        print('new_star_vec error, k=%d'%(k))
                        print('data_i:',new_data_i[0:vis_sat_n,:])
                    print('new_star_vec:',new_star_vec.shape)
                    print('new_star_vec:',new_star_vec[vis_time_step,0:vis_sat_n,:])

    return ''

def Write_MissedIntop(fo_missed_intop,epoch,batch_idx,data_i,active_label,pred_val,pred_softmax,pred_logit,top_indices,err_pos_num_intop_idx,mean_err_pos_num_intop ):
    active_label = np.expand_dims(active_label,axis=-1)
    pred_logit = np.expand_dims(pred_logit,axis=-1)
    fusion_var = np.concatenate([ data_i,active_label,pred_logit,pred_val[:,:,:],pred_softmax],axis=-1 )
    #for i in err_pos_num_intop_idx:
    for i in range(4,8):
        fo_missed_intop.write('epoch %d, batch_idx %d, mean_err_pos_num_intop: %f\n'%(epoch,batch_idx,mean_err_pos_num_intop))
        for j in range(fusion_var.shape[1]):
            str_star = '%10.3g'%(j+1)
            str_star += ''.join(['%10.3g'%fusion_var[i,j,k] for k in range(0,fusion_var.shape[2])])
            top_idx = np.where(j==top_indices[1][i])[0]
            if top_idx.shape[0]:
                str_star += '\ttop-%d'%(top_idx[0])
            elif active_label[i,j] == 1:
                str_star += '\tmissed'
            fo_missed_intop.write(str_star+'\n\n')
        fo_missed_intop.write('\n\n')
        fo_missed_intop.flush()
        print('Write_MissedIntop')


def WritePred(fo,fo_clean,pred_val,pred_logits,cur_data,cur_label):
    assert cur_data.shape[1] == cur_label.shape[1] == pred_val.shape[1]
    cur_label = np.expand_dims(cur_label,axis=-1)
    pred_logits = np.expand_dims(pred_logits,axis=-1)
    fusion_data = np.concatenate([cur_data,pred_val,pred_logits,cur_label],axis=-1).astype(np.float32)
    shape = fusion_data.shape
    for i in range(0,shape[0]):
        TP = FP = T = F = 0
        for j in range(0,shape[1]):
            str_star = '%10.3g'%(j+1)
            str_star += ''.join(['%10.3g'%fusion_data[i,j,k] for k in range(0,shape[2])])
            fo_clean.write(str_star+'\n')
            if pred_logits[i,j,0] != cur_label[i,j,0]:
                str_star += '\terr'
            fo.write(str_star+'\n')

            T += (cur_label[i,j,0] == pred_logits[i,j,0])
            F += (cur_label[i,j,0] != pred_logits[i,j,0])
            TP += (cur_label[i,j,0] == 1) and (pred_logits[i,j,0] == 1)
            FP += (cur_label[i,j,0] == 0) and (pred_logits[i,j,0] == 1)
            TP_FN_str = '%90s'%( 'T:%d   F:%d   TP:%d   FP:%d'%(T,F,TP,FP) )
        fo.write(TP_FN_str+'\n\n')
    fo_clean.flush()
    fo.flush()


def add_data_to_alignbatch_size(data,batch_size):
    n = batch_size - data.shape[0] % batch_size
    for i in range(n):
        data = np.insert(data,data.shape[0],data[-1,:],axis=0)
    return data
def gen_topindices_shuffle_orders(batch_size):
    topindices_shuffle_orders = []
    for k,ntop in enumerate(FLAGS.ntop_candi):
        if k==0: continue
        topindices_shuffle_order_k = np.arange(ntop).astype(np.int32)
        topindices_shuffle_order_k = np.tile(topindices_shuffle_order_k,(batch_size,1))
        if FLAGS.IsShuffleSats:
            for i in range(batch_size):
                np.random.shuffle(topindices_shuffle_order_k[i,:])
        topindices_shuffle_orders.append(topindices_shuffle_order_k)
    return  tuple( topindices_shuffle_orders )


def eval_one_epoch(sess, ops, test_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False

    log_string('----')
    current_data = test_data[:,0:NUM_STAR,:]
    current_label = np.squeeze(test_label[:,0:NUM_STAR])
    raw_N = current_data.shape[0]
    current_data = add_data_to_alignbatch_size(current_data,BATCH_SIZE)
    current_label = add_data_to_alignbatch_size(current_label,BATCH_SIZE)

    num_batches = current_data.shape[0] // BATCH_SIZE

    if FLAGS.only_evaluate:
        if FLAGS.fix_pn:
            predfo = open(os.path.join(LOG_DIR,'eval_pred_fixpn.txt'),'w')
            predfo_clean = open(os.path.join(LOG_DIR,'eval_pred_clean_fixpn.txt'),'w')
        else:
            predfo = open(os.path.join(LOG_DIR,'eval_pred.txt'),'w')
            predfo_clean = open(os.path.join(LOG_DIR,'eval_pred_clean.txt'),'w')
        all_eles =  ['t']+[e for  e in  FLAGS.feed_star_elements]
        all_eles += ['ps0','ps1','pl','l']
        head_str = ''.join(['%9s'%s for s in all_eles])
        predfo.write(head_str+'\n\n')
    batch_idx = -1
    t_per_timestep = -1
    pred_val_ls = []
    pred_logits_ls = []


    total_correct = 0
    total_seen = 0
    loss_sum = np.zeros(shape=(3),dtype=np.float)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]
    num_TP_TN_FN_FP = np.array([0,0,0,0])
    times_errn_histogram = np.zeros(shape=TimeStep_ERR_HIST_N)
    TP_err_histogram = np.zeros(shape=TP_ERR_HIST_N+1)
    num_pos_err_sum = 0
    mean_err_pos_num_intop_sum = 0

    for batch_idx in range(num_batches):
        t0 = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        data_i = current_data[start_idx:end_idx, :, :]
        label_i = current_label[start_idx:end_idx]
        data_i,label_i = shuffle_sats(data_i,label_i)
        sample_loss_weight_i = get_sample_loss_weights(label_i,data_i)

        feed_dict = {ops['satelites_pl']: data_i,
                     ops['labels_pl']: label_i,
                     ops['sampleweights_pl']:sample_loss_weight_i,
                     ops['is_training_pl']: is_training,
                     ops['topindices_shuffle_orders']:gen_topindices_shuffle_orders(BATCH_SIZE) }
        summary, step, loss_val, cascade_preds,cascade_preds_softmax,cascade_labels,loss_classfication,loss_num_pos,top_indices_ls,ntop_pred_sorted_ls =\
            sess.run([ops['merged'], ops['step'], ops['loss'], ops['cascade_preds'],ops['debug_model']['cascade_preds_softmax'],ops['cascade_labels'],
                      ops['loss_classfication'],ops['loss_num_pos'],ops['top_indices_ls'],ops['debug_model']['ntop_pred_sorted_ls']],
                      feed_dict=feed_dict)

        only_ntop_star_vec_ls,is_training_ls = sess.run(
            [ops['debug_model']['only_ntop_star_vec_ls'],
            ops['debug_model']['is_training_ls']],
                                        feed_dict=feed_dict)

        if not FLAGS.fix_pn:
            cascade_pred_logits = [predval_2_predlogit(pred_val) for pred_val in cascade_preds]
        else:
            cascade_pred_logits = []
            for i in range(len(cascade_preds_softmax)):
                cascade_pred_logits.append( predval_2_predlogit(cascade_preds_softmax[i],cascade_labels[i]) )
        #******************************** debug info
        if batch_idx==num_batches-1 or (FLAGS.only_evaluate and batch_idx % 10==0):
            if ISDEBUG:
                vis_sat_n = -1
                last_data_i = data_i[0,:,:]
                print('is_training_ls: ',is_training_ls)
                for k in range(CASCADE_STEP+1):
                    print('\n***\ncascade %d, ntop=%d'%(k,FLAGS.ntop_candi[k]))
                    if k==0 or k< CASCADE_STEP:
                        print('next ntop: %d'%(FLAGS.ntop_candi[k+1]))
                        top_indices = top_indices_ls[k][1]
                        print('ntop indicates: ',top_indices[0,:])
                        print('ntop pred: ',ntop_pred_sorted_ls[k][0,:])

                        new_star_vec = only_ntop_star_vec_ls[k]
                        new_data_i = last_data_i[top_indices[0,:],:]
                        last_data_i = new_data_i
                        print('new_star_vec:',new_star_vec.shape)
                        print('new_star_vec:',new_star_vec[0,0:vis_sat_n,0:1])
                        try:
                            assert (new_star_vec[0,:,:] == new_data_i).all()
                            print('new_star_vec right, k=%d'%(k))
                        except:
                            print('new_star_vec error, k=%d'%(k))
                            print('data_i:',new_data_i[0:vis_sat_n,0:1])

            for m in range(1,len(cascade_labels)):
                print('batch %d, check cascade %d'%(batch_idx,m))
                if m <= 1:
                    check_ntop_data_label(data_i,label_i,only_ntop_star_vec_ls[m-1],cascade_labels[m],top_indices_ls[m-1],cascade_preds[m])
                else:
                    check_ntop_data_label(only_ntop_star_vec_ls[m-2],cascade_labels[m-1],only_ntop_star_vec_ls[m-1],cascade_labels[m],top_indices_ls[m-1],cascade_preds[m])

        #******************************* evaluate metrics
        for m in range(len(cascade_labels)+1):
            #if m != CASCADE_STEP and m!=len(cascade_labels):
            if m != CASCADE_STEP:
                continue


            if m == len(cascade_labels):
                k = m-1
                label_miss_num = np.sum(label_i) - np.sum(cascade_labels[k])
            else:
                k = m
                label_miss_num = 0
            pred_val = cascade_preds[k]
            predsoft_val = cascade_preds_softmax[k]
            pred_logits = cascade_pred_logits[k]
            active_label = cascade_labels[k]

            if test_writer != None:
                test_writer.add_summary(summary, step)
            else:
                if k==0:
                    input_k = data_i
                else:
                    input_k = only_ntop_star_vec_ls[m-1]
                WritePred(predfo,predfo_clean,predsoft_val,pred_logits,
                        input_k,active_label)
            num_pos_label = np.sum(active_label==POSITIVE_LABEL,axis=-1)

            if FLAGS.only_evaluate and IsSaveRes and m==0:
                pred_val_ls.append(pred_val)
                pred_logits_ls.append(pred_logits)

            num_TP_TN_FN_FP_i,num_pos_pred,times_errn_histogram_i,TP_err_histogram_i = get_TPFN(
                pred_logits,active_label,label_miss_num)
            num_TP_TN_FN_FP += num_TP_TN_FN_FP_i
            times_errn_histogram += times_errn_histogram_i
            TP_err_histogram += TP_err_histogram_i

            num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
            num_pos_err_sum += num_pos_err

            if k < len(cascade_labels)-1:
                mean_err_pos_num_intop,ntop_k,err_pos_num_intop_idx = get_ntop_pos_num_err_str(top_indices_ls[k],active_label)
                if FLAGS.only_evaluate and (k == 0 and epoch>=20 and mean_err_pos_num_intop>0.00 and mean_err_pos_num_intop<=0.06):
                    Write_MissedIntop(fo_missed_intop,epoch,batch_idx,data_i,active_label,cascade_preds[k],cascade_preds_softmax[k],pred_logits,top_indices_ls[k],err_pos_num_intop_idx,mean_err_pos_num_intop )
            else:
                mean_err_pos_num_intop = None
                ntop_k = None
            mean_err_pos_num_intop_sum += mean_err_pos_num_intop

            total_seen += (BATCH_SIZE*NUM_STAR)
            loss_sum += np.array([loss_val,loss_classfication,loss_num_pos])
            for i in range(active_label.shape[0]):
                for j in range(active_label.shape[1]):
                    l = active_label[i, j]
                    total_seen_class[l] += 1
            t_per_timestep = (time.time()-t0)/BATCH_SIZE

        if batch_idx==num_batches-1 or (FLAGS.only_evaluate and batch_idx % 10==0):
                eval_logstr = dolog('eval',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum/(1.0+batch_idx),
                                    loss_sum,t_per_timestep,times_errn_histogram,TP_err_histogram,mean_err_pos_num_intop_sum/(1.0+batch_idx),ntop_k,k)
                if False:
                    if m==0 or m<CASCADE_STEP:
                        log_string('ntop pred  : %s'%(np.array_str(ntop_pred_sorted_ls[k][0,0:16],precision=2)))
                    else:
                        log_string('raw pred  : %s'%(np.array_str(cascade_preds[k][0,0:16,1],precision=2)))

    if FLAGS.only_evaluate and False:
        predfo.write('\n'+eval_logstr)
        predfo.close()

        if  IsSaveRes:
            current_label = label_i[0:raw_N+1,:]
            pred_val_all = np.concatenate(pred_val_ls,axis=0)[0:raw_N+1,:]
            pred_logits_all = np.concatenate(pred_logits_ls,axis=0)[0:raw_N+1,:]
            label_errcondition = gen_errcondition_label(current_label,pred_logits_all)
            np.save(os.path.join(LOG_DIR,'pred_val_all'),pred_val_all)
            if FLAGS.fix_pn:
                np.save(os.path.join(LOG_DIR,'pred_logits_all_fixpn'),pred_logits_all)
                np.save(os.path.join(LOG_DIR,'label_errcondition_fixpn'),label_errcondition)
            else:
                np.save(os.path.join(LOG_DIR,'pred_logits_all'),pred_logits_all)
                np.save(os.path.join(LOG_DIR,'label_errcondition_nofixpn'),label_errcondition)
            np.save(os.path.join(LOG_DIR,'test_label'),current_label)
            np.save(os.path.join(LOG_DIR,'test_data'),current_data[0:raw_N+1,:])

            #PlotPred(pred_val_all,pred_logits_all,current_data[0:raw_N+1,:],current_label)
    return eval_logstr

def check_ntop_data_label(raw_data_i,raw_label_i,ntop_data,ntop_label,ntop_indices,ntop_pred):
    batch_size = raw_data_i.shape[0]
    for b in range(batch_size):
        for ntop_i,raw_i in enumerate(ntop_indices[1][b,:]):
            IsDataMatch = (raw_data_i[b,raw_i] == ntop_data[b,ntop_i]).all()
            IsLabelMatch = (raw_label_i[b,raw_i] == ntop_label[b,ntop_i]).all()
            if not IsDataMatch:
                print(raw_data_i[0,:,0])
                print('\n\n')
                print(ntop_data[0,:,0])
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                print('data not match')
            if not IsLabelMatch:
                import pdb; pdb.set_trace()  # XXX BREAKPOINT
                print('label not match')
        #print('both data and label matches, b=',b)
    print('both data and label matches')

def gen_errcondition_label(label,pred_logits):
    label_errcondition = np.zeros_like(label)
    label_errcondition += (label > pred_logits) *1
    label_errcondition += (label < pred_logits) *2
    return label_errcondition


def PlotSavedPred():
    pred_logits_all = np.load(os.path.join(LOG_DIR,'pred_logits_all.npy'))
    pred_val_all = np.load(os.path.join(LOG_DIR,'pred_val_all.npy'))
    test_label_ = np.load(os.path.join(LOG_DIR,'test_label.npy'))
    test_data_ = np.load(os.path.join(LOG_DIR,'test_data.npy'))
    PlotPred(pred_val_all,pred_logits_all,test_data_,test_label_)

def PlotPred(pred_val_all,pred_logits_all,test_data_,test_label_):
    images_path = os.path.join(LOG_DIR,'pred_images')
    if not os.path.exists(images_path):
        os.makedirs(images_path)

    plt.close('all')
    satelite_idx = range(test_label_.shape[1])
    for idx in satelite_idx:
        if idx != 27:
            continue
        t = np.array(range(test_data_.shape[0]))
        isvalid = test_data_[:,idx,0]!=0
        iscorrect = test_label_[:,idx] == pred_logits_all[:,idx]
        err1 = test_label_[:,idx] > pred_logits_all[:,idx]
        err2 = test_label_[:,idx] < pred_logits_all[:,idx]
        valid_correct = isvalid * iscorrect
        valid_err1 = isvalid * err1
        valid_err2 = isvalid * err2

        pred_val =  pred_val_all[:,idx,:]
        fig,ax = plt.subplots(pred_val.shape[1],sharex=True)
        for j in range(pred_val.shape[1]):
            ax[j].plot(t[valid_correct],pred_val[valid_correct,j],'.',label='correct')
            ax[j].plot(t[valid_err1],pred_val[valid_err1,j],'.',label='err1')
            ax[j].plot(t[valid_err2],pred_val[valid_err2,j],'.',label='err2')
            ax[j].set_title('satelite %d - score %d'%(idx,j))
            plt.legend(loc='upper right')
        plt.savefig(images_path+'/satelite %d'%(idx))
    plt.show()

def main():
    train()
    LOG_FOUT.close()

if __name__ == "__main__":
    main()
    '''
    if not FLAGS.plotsaved:
        try:
                main()
        except:
            type, value, tb = sys.exc_info()
            traceback.print_exc()
            pdb.post_mortem(tb)
    else:
        PlotSavedPred()
    '''
