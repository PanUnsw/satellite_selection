#!/bin/bash

train_script=train_cascade.py

N9_WGDOP_B48="python train.py --only_evaluate  --model_epoch 33 --log_dir log_N9_WGDOP_B48_xyzhd_b48_9_mc101 --data_source data_WGDOP --num_pos_ls 9 --feed_star_elements xyzhd"


N9_WGDOP_B128_0="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_mc101-101-101_data_WGDOP_ntop18-10 --ntop_candi 18-10 --cascade_step 0  --neg_loss_w 0.25-0.65 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 101-101-101 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd "
N9_WGDOP_B128_1="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_mc101-101-101_data_WGDOP_ntop18-10 --ntop_candi 18-10 --cascade_step 1  --neg_loss_w 0.25-0.65 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 101-101-101 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd "

N9_WGDOP_B128_0="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_mc101-101_data_WGDOP_ntop12  --ntop_candi 12  --cascade_step 0  --neg_loss_w 1.05-1.05-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 101-101  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd "
N9_WGDOP_B128_0="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_mc101-101-101_data_WGDOP_ntop15-12 --ntop_candi 15-12 --cascade_step 0  --neg_loss_w 0.45-1.05-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 101-101-101 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd "

N9_WGDOP_B128_0_Mcon_0="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop20-10_mc5-3-5MtiCon --ntop_candi 20-10  --cascade_step 0  --neg_loss_w 0.75-1.09-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop20-10_mc5-3-5MtiCon --ntop_candi 20-10  --cascade_step 1  --neg_loss_w 0.75-1.1-1.25 --model_epoch 0-36 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0
#$N9_WGDOP_B128_0_Mcon_1


N9_WGDOP_B128_0_Mcon_0_1610="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-10_mc5-3-5MtiCon --ntop_candi 16-10  --cascade_step 0  --neg_loss_w 0.75-1.1-1.25 --model_epoch 24 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1610="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-10_mc5-3-5MtiCon --ntop_candi 16-10  --cascade_step 1  --neg_loss_w 0.75-1.1-1.25 --model_epoch 24-90 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0_1610
#$N9_WGDOP_B128_0_Mcon_1_1610

N9_WGDOP_B128_0_Mcon_0_1510="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop15-10_mc5-3-8MtiCon --ntop_candi 15-10  --cascade_step 0  --neg_loss_w 0.75-1.1-1.25 --model_epoch 10 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-8  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1510="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop15-10_mc5-3-8MtiCon --ntop_candi 15-10  --cascade_step 1  --neg_loss_w 0.75-1.1-1.25 --model_epoch 10-84 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-8  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"


#$N9_WGDOP_B128_0_Mcon_0_1510
$N9_WGDOP_B128_0_Mcon_1_1510
