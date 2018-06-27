import time
import pdb, traceback
import tensorflow as tf
import argparse
import math
import h5py
import numpy as np
from reader import Star_Reader
from sklearn.preprocessing import OneHotEncoder
from model import get_model,get_loss,placeholder_inputs,n_argmax

import matplotlib.pyplot as plt
import os,sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)



parser = argparse.ArgumentParser()

parser.add_argument('--a', default='', help='the name of configuration')

parser.add_argument('--feed_star_elements', default='xyzhd', help='part of xyzhdgn')
parser.add_argument('--batch_size', type=int, default=64, help='Batch Size during training [default: 24]')
parser.add_argument('--neg_loss_w',default=0.7,help='loss weight for negative sample')
parser.add_argument('--empty_loss_w',default=0.01,help='loss weight for empty sample')
parser.add_argument('--loss_w_class', type=float, default=1.0, help='The loss weight for classification')
parser.add_argument('--loss_w_numpos', type=float, default=0.0, help='The loss weight for positive number constrain')
parser.add_argument('--num_pos_ls',default='9',help='7-9-11-13 positive num list of data to be selected')
parser.add_argument('--model_config',type=int,default=101,help='model config: 101, 6, 7')
parser.add_argument('--data_source',default='data_WGDOP_small',help='data_sync or data_withg')

parser.add_argument('--UseErrCondLabel',action='store_true',help='UseErrCondLabel')
parser.add_argument('--UseEmptyLabel',action='store_true',help='Use empty as a seperate label')
parser.add_argument('--IsRegression',action='store_true',help='Use positve satelite index as label')
parser.add_argument('--IsHingeloss',action='store_true',help='Use positve satelite index as label')

parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=4, help='Epoch to run [default: 50]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=200000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.7, help='Decay rate for lr decay [default: 0.5]')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--finetune',action='store_true',help='finetune')
parser.add_argument('--fix_pn',action='store_true',help='constrain the positive number forcely')
parser.add_argument('--model_epoch', type=int, default=3, help='The epoch of model to use')
parser.add_argument('--plotsaved',action='store_true',help='only plotsaved')
FLAGS = parser.parse_args()


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
else:
    log_name = 'log_train.txt'
    if 'LOG' in FLAGS.log_dir:
        LOG_DIR = FLAGS.log_dir
    else:
        LOG_DIR = FLAGS.log_dir+FLAGS.a+'_'+FLAGS.feed_star_elements+'_b'+str(BATCH_SIZE)+'_'+FLAGS.num_pos_ls+'_mc'+str(FLAGS.model_config)+'_'+FLAGS.data_source
    if FLAGS.IsHingeloss:
        LOG_DIR += '_HLoss'
LOG_DIR = os.path.join(BASE_DIR+'/RES',LOG_DIR)
FLAGS.model_path = os.path.join(LOG_DIR,'model.ckpt-'+str(FLAGS.model_epoch))
MODEL_PATH = FLAGS.model_path

if not os.path.exists(LOG_DIR): os.makedirs(LOG_DIR)
if not FLAGS.only_evaluate:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
else:
    LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'a')

FUSION_LOG_FOUT = open(os.path.join(BASE_DIR+'/RES', 'fusion_log.txt'), 'a')

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
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99



# Load data
if FLAGS.only_evaluate:
    training_data_rate = 0.0
else:
    training_data_rate = 0.7
IsSaveRes = True
NUM_STAR = Star_Reader.max_num_instars
star_reader = Star_Reader(train_rate=training_data_rate,data_source=FLAGS.data_source,
                          IsOnlyEval=FLAGS.only_evaluate,IsUseErrCondLabel=FLAGS.UseErrCondLabel,
                          IsEmptyLabel=FLAGS.UseEmptyLabel,IsRegression=FLAGS.IsRegression)
NUM_CHANNEL = len(FLAGS.feed_star_elements)
train_data,train_label, test_data,test_label = star_reader.get_train_test(
    feed_star_elements=FLAGS.feed_star_elements,num_pos_ls=FLAGS.num_pos_ls)


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




def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
log_string(str(FLAGS)+'\n')
log_string(star_reader.data_summary_str)

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
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            satelites_pl, labels_pl, sampleweights_pl = placeholder_inputs(BATCH_SIZE, NUM_STAR, NUM_CHANNEL,model_config=FLAGS.model_config,num_class=NUM_CLASSES)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            if FLAGS.IsHingeloss:
                model_num_class = NUM_CLASSES - 1
            else:
                model_num_class = NUM_CLASSES
            pred = get_model(satelites_pl, is_training_pl, bn_decay=bn_decay,model_config=FLAGS.model_config,num_class=model_num_class )
            loss,loss_classfication,loss_num_pos = get_loss(pred, labels_pl,sampleweights_pl,
                                                            [FLAGS.loss_w_class,FLAGS.loss_w_numpos],num_class=model_num_class)
            tf.summary.scalar('loss', loss)
            tf.summary.histogram('pred_hist',pred)

            if FLAGS.IsRegression:
                correct = tf.equal(tf.to_int64(pred), tf.to_int64(labels_pl))
            else:
                correct = tf.equal(tf.argmax(pred, 2), tf.to_int64(labels_pl))
            accuracy = tf.reduce_sum(tf.cast(correct, tf.float32)) / float(BATCH_SIZE*NUM_STAR)
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
            saver = tf.train.Saver(max_to_keep=50)

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
               'pred': pred,
               'loss': loss,
               'loss_classfication': loss_classfication,
               'loss_num_pos':loss_num_pos,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        start_epoch = 0
        if FLAGS.finetune:
            start_epoch += FLAGS.model_epoch+1
        for epoch in range(start_epoch,start_epoch+MAX_EPOCH):
            log_string('\n**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()
            if FLAGS.finetune:
                saver.restore(sess,MODEL_PATH)
                log_string('fine tune, restore model from: \n\t%s'%(MODEL_PATH))
            if not FLAGS.only_evaluate:
                train_log_str = train_one_epoch(sess, ops, train_writer,epoch)
            else:
                saver.restore(sess,MODEL_PATH)
                log_string('restore model from: \n\t%s'%(MODEL_PATH))
                train_log_str=''
            eval_log_str = eval_one_epoch(sess, ops, test_writer,epoch)

            # Save the variables to disk.
            if ( epoch >= 0 and (epoch) % 2 == 0 ) and (not FLAGS.only_evaluate):
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                log_string("Model saved in file: %s" % save_path)
            if epoch == MAX_EPOCH-1 and not FLAGS.only_evaluate:
                FUSION_LOG_FOUT.write(str(FLAGS)+'\n\n'+train_log_str+'\n'+eval_log_str+'\n\n')


TimeStep_ERR_HIST_N = 8
TP_ERR_HIST_N = 4
def get_TPFN(pred_logits,label):
        Pred_True = (pred_logits == label)
        Pred_Pos =  (pred_logits == POSITIVE_LABEL)
        TP = Pred_Pos * Pred_True
        TN = Pred_True * (1 - Pred_Pos)
        FN = (1 - Pred_True) * (1 - Pred_Pos)
        FP = (1 - Pred_True) * Pred_Pos
        num_TP_TN_FN_FP = np.array( [ np.sum(TP), np.sum(TN),np.sum(FN), np.sum(FP)])
        pos_num = np.sum(Pred_Pos==POSITIVE_LABEL,axis=1)
        correct = np.sum(Pred_True)

        # the err time steps number histogram
        pred_errnum = np.sum(Pred_True==False,axis=1)
        times_errn_histogram = np.histogram(pred_errnum,bins=range(TimeStep_ERR_HIST_N+1))[0]

        # the err satelite number histogram
        TP_err_histogram = np.histogram( np.sum(TP,axis=1)-np.sum(label==POSITIVE_LABEL,axis=1),bins=range(-TP_ERR_HIST_N,2) )[0]
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

def dolog(tot,epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,
          t_per_timestep,times_errn_histogram,TP_err_histogram):
    if FLAGS.IsRegression:
        loss_str = np.array_str(loss_sum[0]/(batch_idx+1),precision=3)
        log_string(loss_str)
        return loss_str

    logstr = ''
    recall, precision,correct = cal_accu(num_TP_TN_FN_FP)
    ave_num_pos_err =  num_pos_err_sum/float(batch_idx+1)/float(BATCH_SIZE)
    loss_str = np.array_str(loss_sum/(batch_idx+1),precision=3)
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
                    slw[i,j] = FLAGS.neg_loss_w
    return slw

def shuffle_train(train_data,train_label):
    N = train_data.shape[0]
    idxs = np.arange(N)
    np.random.shuffle(idxs)
    train_data = train_data[idxs,:,:]
    train_label = train_label[idxs,:]
    return train_data,train_label

def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_STAR,:], train_label[:,0:NUM_STAR])
    current_data = train_data
    current_label = train_label
    current_data,current_label = shuffle_train(current_data,current_label)

    num_batches = current_data.shape[0] // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = np.zeros(shape=(3),dtype=np.float)

    num_TP_TN_FN_FP = np.array([0,0,0,0])
    times_errn_histogram = np.zeros(shape=TimeStep_ERR_HIST_N)
    TP_err_histogram = np.zeros(shape=TP_ERR_HIST_N+1)
    num_pos_err_sum = 0
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
        sample_loss_weight_i = get_sample_loss_weights(label_i,data_i)

        feed_dict = {ops['satelites_pl']: data_i,
                     ops['labels_pl']: label_i,
                     ops['sampleweights_pl']:sample_loss_weight_i,
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val,loss_classfication,loss_num_pos = \
            sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred'],
                        ops['loss_classfication'],ops['loss_num_pos']], feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        if  FLAGS.IsRegression:
            pred_logit = pred_val.astype(np.int)
        elif FLAGS.IsHingeloss:
            pred_logit = np.squeeze((pred_val>0.5).astype(np.int),-1)
        else:
            pred_logit = np.argmax(pred_val, 2)


        correct = np.sum(pred_logit == current_label[start_idx:end_idx])

        num_TP_TN_FN_FP_i,num_pos_pred,times_errn_histogram_i,TP_err_histogram_i = get_TPFN(pred_logit,current_label[start_idx:end_idx])
        num_TP_TN_FN_FP = num_TP_TN_FN_FP_i
        times_errn_histogram += times_errn_histogram_i
        TP_err_histogram += TP_err_histogram_i

        num_pos_label = np.sum(current_label[start_idx:end_idx]==POSITIVE_LABEL,axis=-1)
        num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
        num_pos_err_sum += num_pos_err

        total_seen += (BATCH_SIZE*NUM_STAR)
        loss_sum += np.array([loss_val,loss_classfication,loss_num_pos])
        t_per_timestep = (time.time()-t0)/BATCH_SIZE
        if (epoch == 0 and batch_idx <= 10) or (batch_idx>0 and batch_idx%100==0):
            dolog('train',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,t_per_timestep,times_errn_histogram,TP_err_histogram)
            if  FLAGS.IsRegression or FLAGS.IsHingeloss:
                print('pred  : %s'%(np.array_str(pred_val[0,0:10,0],precision=2)))
                print('label : %s\n'%(np.array_str(label_i[0,0:10],precision=2)))
    return dolog('train',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,t_per_timestep,times_errn_histogram,TP_err_histogram)

def WritePred(fo,fo_clean,pred_val,pred_logits,cur_data,cur_label):
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

def eval_one_epoch(sess, ops, test_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = np.zeros(shape=(3),dtype=np.float)
    total_seen_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:,0:NUM_STAR,:]
    current_label = np.squeeze(test_label[:,0:NUM_STAR])
    raw_N = current_data.shape[0]
    current_data = add_data_to_alignbatch_size(current_data,BATCH_SIZE)
    current_label = add_data_to_alignbatch_size(current_label,BATCH_SIZE)

    num_batches = current_data.shape[0] // BATCH_SIZE

    num_TP_TN_FN_FP = np.array([0,0,0,0])
    times_errn_histogram = np.zeros(shape=TimeStep_ERR_HIST_N)
    TP_err_histogram = np.zeros(shape=TP_ERR_HIST_N+1)
    num_pos_err_sum = 0

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
    for batch_idx in range(num_batches):
        t0 = time.time()
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE
        data_i = current_data[start_idx:end_idx, :, :]
        label_i = current_label[start_idx:end_idx]
        sample_loss_weight_i = get_sample_loss_weights(label_i,data_i)

        feed_dict = {ops['satelites_pl']: data_i,
                     ops['labels_pl']: label_i,
                     ops['sampleweights_pl']:sample_loss_weight_i,
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val,loss_classfication,loss_num_pos =\
            sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred'],
                      ops['loss_classfication'],ops['loss_num_pos']], feed_dict=feed_dict)

        if  FLAGS.IsRegression:
            pred_logits = pred_val.astype(np.int)
        elif FLAGS.IsHingeloss:
            pred_logits = np.squeeze((pred_val>0.5).astype(np.int),-1)
        else:
            if FLAGS.fix_pn:
                pred_logits = n_argmax(pred_val, current_label[start_idx:end_idx],current_data[start_idx:end_idx, :, :])
            else:
                pred_logits = np.argmax(pred_val, 2)

        if test_writer != None:
            test_writer.add_summary(summary, step)
        else:
            WritePred(predfo,predfo_clean,pred_val,pred_logits,current_data[start_idx:end_idx, :, :],current_label[start_idx:end_idx])
        num_pos_label = np.sum(current_label[start_idx:end_idx]==POSITIVE_LABEL,axis=-1)

        if FLAGS.only_evaluate and IsSaveRes:
            pred_val_ls.append(pred_val)
            pred_logits_ls.append(pred_logits)

        num_TP_TN_FN_FP_i,num_pos_pred,times_errn_histogram_i,TP_err_histogram_i = get_TPFN(pred_logits,current_label[start_idx:end_idx])
        num_TP_TN_FN_FP += num_TP_TN_FN_FP_i
        times_errn_histogram += times_errn_histogram_i
        TP_err_histogram += TP_err_histogram_i

        num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
        num_pos_err_sum += num_pos_err

        total_seen += (BATCH_SIZE*NUM_STAR)
        loss_sum += np.array([loss_val,loss_classfication,loss_num_pos])
        for i in range(start_idx, end_idx):
            for j in range(NUM_STAR):
                l = current_label[i, j]
                total_seen_class[l] += 1
        t_per_timestep = (time.time()-t0)/BATCH_SIZE
        if FLAGS.only_evaluate and batch_idx % 100==0:
            eval_logstr = dolog('eval',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,t_per_timestep,times_errn_histogram,TP_err_histogram)

    eval_logstr = dolog('eval',epoch,batch_idx,num_TP_TN_FN_FP,num_pos_err_sum,loss_sum,t_per_timestep,times_errn_histogram,TP_err_histogram)

    if FLAGS.only_evaluate:
        predfo.write('\n'+eval_logstr)
        predfo.close()

        if  IsSaveRes:
            current_label = current_label[0:raw_N+1,:]
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

            PlotPred(pred_val_all,pred_logits_all,current_data[0:raw_N+1,:],current_label)
    return eval_logstr

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
