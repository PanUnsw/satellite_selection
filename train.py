import tensorflow as tf
import argparse
import math
import h5py
import numpy as np
from reader import Star_Reader
from sklearn.preprocessing import OneHotEncoder
from model import get_model,get_loss,placeholder_inputs

import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(BASE_DIR)



parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--max_epoch', type=int, default=20, help='Epoch to run [default: 50]')
parser.add_argument('--batch_size', type=int, default=32, help='Batch Size during training [default: 24]')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Initial learning rate [default: 0.001]')
parser.add_argument('--momentum', type=float, default=0.9, help='Initial learning rate [default: 0.9]')
parser.add_argument('--optimizer', default='adam', help='adam or momentum [default: adam]')
parser.add_argument('--decay_step', type=int, default=300000, help='Decay step for lr decay [default: 300000]')
parser.add_argument('--decay_rate', type=float, default=0.5, help='Decay rate for lr decay [default: 0.5]')

parser.add_argument('--only_evaluate',action='store_true',help='do not train')
parser.add_argument('--model_epoch', type=int, default=20, help='The epoch of model to use')
FLAGS = parser.parse_args()


BATCH_SIZE = FLAGS.batch_size
MAX_EPOCH = FLAGS.max_epoch
BASE_LEARNING_RATE = FLAGS.learning_rate
GPU_INDEX = FLAGS.gpu
MOMENTUM = FLAGS.momentum
OPTIMIZER = FLAGS.optimizer
DECAY_STEP = FLAGS.decay_step
DECAY_RATE = FLAGS.decay_rate

FLAGS.model_path = os.path.join(FLAGS.log_dir,'model.ckpt-'+str(FLAGS.model_epoch))
MODEL_PATH = FLAGS.model_path

if FLAGS.only_evaluate:
    MAX_EPOCH = 1
    log_name = 'log_eval.txt'
else:
    log_name = 'log_train.txt'



LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
LOG_FOUT = open(os.path.join(LOG_DIR, log_name), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

NUM_CLASSES = 2

BN_INIT_DECAY = 0.5
BN_DECAY_DECAY_RATE = 0.5
#BN_DECAY_DECAY_STEP = float(DECAY_STEP * 2)
BN_DECAY_DECAY_STEP = float(DECAY_STEP)
BN_DECAY_CLIP = 0.99



# Load data
NUM_STAR = Star_Reader.max_num_instars
NUM_CHANNEL = Star_Reader.input_channel

star_reader = Star_Reader()
train_data,train_label, test_data,test_label = star_reader.get_train_test('org',0.7)


def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)
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
            stars_pl, labels_pl = placeholder_inputs(BATCH_SIZE, NUM_STAR, NUM_CHANNEL)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            # Note the global_step=batch parameter to minimize.
            # That tells the optimizer to helpfully increment the 'batch' parameter for you every time it trains.
            batch = tf.Variable(0)
            bn_decay = get_bn_decay(batch)
            tf.summary.scalar('bn_decay', bn_decay)

            # Get model and loss
            pred = get_model(stars_pl, is_training_pl, bn_decay=bn_decay)
            loss,loss_classfication,loss_num_pos = get_loss(pred, labels_pl)
            tf.summary.scalar('loss', loss)

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
            saver = tf.train.Saver()

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

        ops = {'stars_pl': stars_pl,
               'labels_pl': labels_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss,
               'loss_num_pos':loss_num_pos,
               'train_op': train_op,
               'merged': merged,
               'step': batch}

        for epoch in range(MAX_EPOCH):
            log_string('\n**** EPOCH %03d ****' % (epoch))
            sys.stdout.flush()

            if not FLAGS.only_evaluate:
                train_one_epoch(sess, ops, train_writer,epoch)
            else:
                saver.restore(sess,MODEL_PATH)
                log_string('restore model from: \n\t%s'%(MODEL_PATH))
            eval_one_epoch(sess, ops, test_writer)

            # Save the variables to disk.
            if ( epoch % 4 == 0 ) and (not FLAGS.only_evaluate):
                save_path = saver.save(sess, os.path.join(LOG_DIR, "model.ckpt"),global_step=epoch)
                log_string("Model saved in file: %s" % save_path)


def get_TPFN(pred_logits,label):
        Pred_True = (pred_logits == label)
        Pred_Pos =  (pred_logits == 1)
        TP = Pred_Pos * Pred_True
        TN = Pred_True * (1 - Pred_Pos)
        FN = (1 - Pred_True) * (1 - Pred_Pos)
        FP = (1 - Pred_True) * Pred_Pos
        num_TP_TN_FN_FP = np.array( [ np.sum(TP), np.sum(TN),np.sum(FN), np.sum(FP)])
        pos_num = np.sum(Pred_Pos,axis=1)
        correct = np.sum(Pred_True)
        return num_TP_TN_FN_FP, pos_num

def cal_accu(num_TP_TN_FN_FP):
    num_TP_TN_FN_FP = num_TP_TN_FN_FP.astype(np.float32)
    P = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[3]
    real_P = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[2]
    recall = num_TP_TN_FN_FP[0] / real_P
    precision = num_TP_TN_FN_FP[0] / P
    T = num_TP_TN_FN_FP[0] + num_TP_TN_FN_FP[1]
    correct = T / np.sum(num_TP_TN_FN_FP)
    return recall, precision,correct

def train_one_epoch(sess, ops, train_writer,epoch):
    """ ops: dict mapping from string to tf ops """
    is_training = True

    log_string('----')
    #current_data, current_label, _ = provider.shuffle_data(train_data[:,0:NUM_STAR,:], train_label[:,0:NUM_STAR])
    current_data = train_data
    current_label = train_label

    num_batches = current_data.shape[0] // BATCH_SIZE

    total_correct = 0
    total_seen = 0
    loss_sum = 0

    num_TP_TN_FN_FP = np.array([0,0,0,0])
    num_pos_err_sum = 0
    #print('total batch num = ',num_batches)
    def log_train():
        recall, precision,correct = cal_accu(num_TP_TN_FN_FP)
        ave_num_pos_err =  num_pos_err_sum/float(batch_idx+1)/float(BATCH_SIZE)
        log_string('epoch %d batch %d    train mean loss: %f    ave_num_pos_err: %f' % (epoch,batch_idx,loss_sum / float(batch_idx+1),ave_num_pos_err))
        log_string('epoch %d batch %d    train recall: %f, precision: %f, accuracy: %f'% (
            epoch, batch_idx, recall,precision,correct))

    for batch_idx in range(num_batches):
        #if batch_idx % 100 == 0:
            #print('Current batch/total batch num: %d/%d'%(batch_idx,num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['stars_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training,}
        summary, step, _, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['train_op'], ops['loss'], ops['pred']],
                                         feed_dict=feed_dict)
        train_writer.add_summary(summary, step)
        pred_val = np.argmax(pred_val, 2)
        correct = np.sum(pred_val == current_label[start_idx:end_idx])

        num_TP_TN_FN_FP_i,num_pos_pred = get_TPFN(pred_val,current_label[start_idx:end_idx])
        num_TP_TN_FN_FP += num_TP_TN_FN_FP_i

        num_pos_label = np.sum(current_label[start_idx:end_idx],axis=-1)
        num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
        num_pos_err_sum += num_pos_err

        total_seen += (BATCH_SIZE*NUM_STAR)
        loss_sum += loss_val

       # print('\t\t\t\t\t\t loss_class, loss_num_pos: %f,  %f, pos_num = %d'%(
       #     loss_val, sess.run(ops['loss_num_pos'],feed_dict=feed_dict),np.mean(pos_num_i) ))

        if (epoch == 0 and batch_idx <= 10) or (batch_idx>0 and batch_idx%100==0):
            log_train()
    log_train()



def eval_one_epoch(sess, ops, test_writer):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    total_correct = 0
    total_seen = 0
    loss_sum = 0
    total_seen_class = [0 for _ in range(NUM_CLASSES)]

    log_string('----')
    current_data = test_data[:,0:NUM_STAR,:]
    current_label = np.squeeze(test_label[:,0:NUM_STAR])

    num_batches = current_data.shape[0] // BATCH_SIZE

    num_TP_TN_FN_FP = np.array([0,0,0,0])
    num_pos_err_sum = 0

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = (batch_idx+1) * BATCH_SIZE

        feed_dict = {ops['stars_pl']: current_data[start_idx:end_idx, :, :],
                     ops['labels_pl']: current_label[start_idx:end_idx],
                     ops['is_training_pl']: is_training}
        summary, step, loss_val, pred_val = sess.run([ops['merged'], ops['step'], ops['loss'], ops['pred']],
                                      feed_dict=feed_dict)
        if test_writer != None:
            test_writer.add_summary(summary, step)
        pred_logits = np.argmax(pred_val, 2)
        num_TP_TN_FN_FP_i,num_pos_pred = get_TPFN(pred_logits,current_label[start_idx:end_idx])
        num_TP_TN_FN_FP += num_TP_TN_FN_FP_i

        num_pos_label = np.sum(current_label[start_idx:end_idx],axis=-1)
        num_pos_err = np.sum(np.absolute( num_pos_label - num_pos_pred ))
        num_pos_err_sum += num_pos_err

        total_seen += (BATCH_SIZE*NUM_STAR)
        loss_sum += (loss_val*BATCH_SIZE)
        for i in range(start_idx, end_idx):
            for j in range(NUM_STAR):
                l = current_label[i, j]
                total_seen_class[l] += 1


      #  print('\t\t\t\t\t\t loss_class, loss_num_pos: %f,  %f, pos_num = %d'%(
      #      loss_val, sess.run(ops['loss_num_pos'],feed_dict=feed_dict),np.mean(pos_num_i) ))
      #  #print(pos_num_i)
      #  mse = ((pos_num_i-11)**2).mean()
      #  print(mse)

    recall, precision,correct = cal_accu(num_TP_TN_FN_FP)
    ave_num_pos_err =  num_pos_err_sum/float(batch_idx+1)/float(BATCH_SIZE)
    log_string('eval mean loss: %f    ave_num_pos_err: %f' % (loss_sum / float(total_seen/NUM_STAR),ave_num_pos_err))
    log_string('eval recall: %f, precision: %f, accuracy: %f'% (recall,precision,correct))
    #log_string('eval class accuracies: %s' % (np.array_str(class_accuracies)))
    #log_string('eval avg class acc: %f' % (np.mean(class_accuracies)))



if __name__ == "__main__":
    train()
    LOG_FOUT.close()
