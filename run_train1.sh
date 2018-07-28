#!/bin/bash

train_script=train_cascade.py


#----------------------------------------------------------------------------------------------------

#---------------- hd
N9_WGDOP_B128_0_Mcon_0_1611_12_hd="python $train_script --ntop_candi 20-13  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --max_epoch 1 --UseMultiConcat --gpu 1"
N9_WGDOP_B128_0_Mcon_1_1611_12_hd="python $train_script --ntop_candi 20-13  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --max_epoch 101 --UseMultiConcat --gpu 1"


#$N9_WGDOP_B128_0_Mcon_0_1611_12_hd
#$N9_WGDOP_B128_0_Mcon_1_1611_12_hd
###use for 12 sats


#----------- xyz
N9_WGDOP_B128_0_Mcon_0_1611_12_xyz="python $train_script --ntop_candi 20-13  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements xyz --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_12_xyz="python $train_script --ntop_candi 20-13  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements xyz --max_epoch 101 --UseMultiConcat"


#$N9_WGDOP_B128_0_Mcon_0_1611_12_xyz
#$N9_WGDOP_B128_0_Mcon_1_1611_12_xyz
###use for 12 sats

#----------- xyzhd
N9_WGDOP_B128_0_Mcon_0_1611_12="python $train_script --ntop_candi 20-13  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --max_epoch 1 --UseMultiConcat --gpu 1"
N9_WGDOP_B128_0_Mcon_1_1611_12="python $train_script --ntop_candi 20-13  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --max_epoch 101 --UseMultiConcat --gpu 1"


#$N9_WGDOP_B128_0_Mcon_0_1611_12
#$N9_WGDOP_B128_0_Mcon_1_1611_12

###use for 12 sats


#----------------------------------------------------------------------------------------------------
#  only xyz
N9_WGDOP_B128_0_Mcon_0_1611_9_XYZ="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyz --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_9_XYZ="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyz --max_epoch 101 --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0_1611_9_XYZ
#$N9_WGDOP_B128_0_Mcon_1_1611_9_XYZ
##use for 9

# --------
#  only hd
N9_WGDOP_B128_0_Mcon_0_1611_9_HD="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements hd --max_epoch 1 --UseMultiConcat --gpu 1"
N9_WGDOP_B128_0_Mcon_1_1611_9_HD="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements hd --max_epoch 101 --UseMultiConcat --gpu 1"

#$N9_WGDOP_B128_0_Mcon_0_1611_9_HD
#$N9_WGDOP_B128_0_Mcon_1_1611_9_HD
##use for 9

# --------
#  only hd with model to PointNet
N9_WGDOP_B128_0_Mcon_0_1611_9_HD="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-5222-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --decay_rate 0.35  --bn_decay_rate 0.35 --batch_size 128 --feed_star_elements hd --max_epoch 1 --UseMultiConcat --gpu 0"
N9_WGDOP_B128_0_Mcon_1_1611_9_HD="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-5222-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --decay_rate 0.35 --bn_decay_rate 0.35 --batch_size 128 --feed_star_elements hd --max_epoch 60 --UseMultiConcat --gpu 0"

$N9_WGDOP_B128_0_Mcon_0_1611_9_HD
$N9_WGDOP_B128_0_Mcon_1_1611_9_HD
##use for 9

# -------xyzhd
N9_WGDOP_B128_0_Mcon_0_1611_9="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_9="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 101 --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0_1611_9
#$N9_WGDOP_B128_0_Mcon_1_1611_9
##use for 9


