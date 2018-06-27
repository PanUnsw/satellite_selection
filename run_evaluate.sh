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


N9_WGDOP_B128_0_Mcon_0_1710="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop17-10_mc5-3-5MtiCon --ntop_candi 17-10  --cascade_step 0  --neg_loss_w 0.75-1.1-1.25 --model_epoch 10 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1710="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop17-10_mc5-3-5MtiCon --ntop_candi 17-10  --cascade_step 1  --neg_loss_w 0.75-1.1-1.25 --model_epoch 10-78 --a N9_WGDOP_B128 --data_source data_WGDOP_test --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0_1710
#$N9_WGDOP_B128_0_Mcon_1_1710


N12_WGDOP_B128_0_Mcon_0_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_new_ntop20-13_mc9-51-8MtiCon --ntop_candi 20-13 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 2    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new --model_config 9-51-8  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_new_ntop20-13_mc9-51-8MtiCon --ntop_candi 20-13 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 2-42 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new --model_config 9-51-8  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611
#$N12_WGDOP_B128_0_Mcon_1_1611


N12_WGDOP_B128_0_Mcon_0_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzh_b128_12_data_WGDOP_12_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test12 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_12_ntop20-13_mc9-53-18MtiCon --ntop_candi 20-13 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-18 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_12 --model_config 9-53-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611
#$N12_WGDOP_B128_0_Mcon_1_1611

##test for 12 sats



#-------------------------------------------------------------------------
N12_WGDOP_B128_0_Mcon_0_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_9_ntop20-11_mc9-53-18MtiCon --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.1-1.25  --model_epoch 0     --a N9_WGDOP_B128  --data_source_test data_WGDOP_test9 --model_config 9-53-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_9_ntop20-11_mc9-53-18MtiCon --ntop_candi 20-11 --cascade_step 1 --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-16     --a N9_WGDOP_B128  --data_source_test data_WGDOP_test_9 --model_config 9-53-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611
#$N12_WGDOP_B128_0_Mcon_1_1611

##test for 9 sats





#-------------------------------------------------------------------------
N12_WGDOP_B128_0_Mcon_0_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-48 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611
$N12_WGDOP_B128_0_Mcon_1_1611


##use for 12 sats


# only use xyz

N12_WGDOP_B128_0_Mcon_0_1611_xyz="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyz_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyz --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611_xyz="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyz_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-48 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements xyz --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611_xyz
$N12_WGDOP_B128_0_Mcon_1_1611_xyz

##use for 12 sats


# only use hd
N12_WGDOP_B128_0_Mcon_0_1611_HD="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_hd_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --UseMultiConcat"
N12_WGDOP_B128_0_Mcon_1_1611_HD="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_hd_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon --ntop_candi 20-13 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-48 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test_new2 --model_config 9-50-18  --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --UseMultiConcat --fix_pn"

#$N12_WGDOP_B128_0_Mcon_0_1611_HD
$N12_WGDOP_B128_0_Mcon_1_1611_HD

##use for 12 sats









#-------------------------------------------------------------------------
N9_WGDOP_B128_0_Mcon_0_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-8 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --UseMultiConcat --fix_pn"


#$N9_WGDOP_B128_0_Mcon_0_1611
#$N9_WGDOP_B128_0_Mcon_1_1611
##use for 9 sats


#-----------
# only use xyz
N9_WGDOP_B128_0_Mcon_0_1611_XYZ="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyz_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyz --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_XYZ="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_xyz_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-46 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyz --UseMultiConcat --fix_pn"


#$N9_WGDOP_B128_0_Mcon_0_1611_XYZ
#$N9_WGDOP_B128_0_Mcon_1_1611_XYZ
##use for 9 sats

#-----------
# only use hd
N9_WGDOP_B128_0_Mcon_0_1611_HD="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_hd_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0    --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements hd --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_HD="python $train_script --only_evaluate --log_dir logN9_WGDOP_B128_hd_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon --ntop_candi 20-11 --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-8 --a N9_WGDOP_B128 --data_source_test data_WGDOP_test2 --model_config 9-52-18  --num_pos_ls 9 --batch_size 128 --feed_star_elements hd --UseMultiConcat --fix_pn"


#$N9_WGDOP_B128_0_Mcon_0_1611_HD
#$N9_WGDOP_B128_0_Mcon_1_1611_HD
##use for 9 sats

