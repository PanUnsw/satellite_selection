#!/bin/bash

train_script=train_cascade.py


N9_WGDOP="python $train_script --a N9_WGDOP --data_source data_WGDOP --num_pos_ls 9 --feed_star_elements xyzhd"



N9_WGDOP_B128_0_Mcon_0_1611="python $train_script --ntop_candi 16-11  --cascade_step 0  --neg_loss_w 1.15-2.1-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP --data_source_test data_WGDOP_test --model_config 9-5-8  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 11 --UseMultiConcat --activation_fn relu"
N9_WGDOP_B128_0_Mcon_1_1611="python $train_script --ntop_candi 16-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 10 --a N9_WGDOP_B128 --data_source data_WGDOP --data_source_test data_WGDOP_test --model_config 9-5-8  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 97 --UseMultiConcat --activation_fn relu"
N9_WGDOP_B128_0_Mcon_1_1611_FT="python $train_script --ntop_candi 16-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 10-34 --a N9_WGDOP_B128 --data_source data_WGDOP --model_config 9-5-8 --data_source_test data_WGDOP_test --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 197 --UseMultiConcat --finetune --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-11_mc9-5-8MtiCon"


#$N9_WGDOP_B128_0_Mcon_1_1611_FT


#$N9_WGDOP_B128_0_Mcon_0_1611
#$N9_WGDOP_B128_0_Mcon_1_1611


#----------------------------------------------------------------------------------------------------
N9_WGDOP_B128_0_Mcon_0_1611_12="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_9 --model_config 9-53-18 --data_source_test data_WGDOP_test9 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_12="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_9 --model_config 9-53-18 --data_source_test data_WGDOP_test9 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 77 --UseMultiConcat"


#$N9_WGDOP_B128_0_Mcon_0_1611_12
#$N9_WGDOP_B128_0_Mcon_1_1611_12
###test for 12



#----------------------------------------------------------------------------------------------------
N9_WGDOP_B128_0_Mcon_0_1611_12="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.1-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_9 --model_config 9-53-18 --data_source_test data_WGDOP_test9 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_12="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_9 --model_config 9-53-18 --data_source_test data_WGDOP_test9 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 77 --UseMultiConcat"


#$N9_WGDOP_B128_0_Mcon_0_1611_12
#$N9_WGDOP_B128_0_Mcon_1_1611_12
###test for 9



#----------------------------------------------------------------------------------------------------






#---------------- hd
N9_WGDOP_B128_0_Mcon_0_1611_12_hd="python $train_script --ntop_candi 20-13  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --max_epoch 1 --UseMultiConcat --gpu 1"
N9_WGDOP_B128_0_Mcon_1_1611_12_hd="python $train_script --ntop_candi 20-13  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new2 --model_config 9-50-18 --data_source_test data_WGDOP_test_new2 --num_pos_ls 12 --batch_size 128 --feed_star_elements hd --max_epoch 101 --UseMultiConcat --gpu 1"


$N9_WGDOP_B128_0_Mcon_0_1611_12_hd
$N9_WGDOP_B128_0_Mcon_1_1611_12_hd
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


$N9_WGDOP_B128_0_Mcon_0_1611_12
$N9_WGDOP_B128_0_Mcon_1_1611_12

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

$N9_WGDOP_B128_0_Mcon_0_1611_9_HD
$N9_WGDOP_B128_0_Mcon_1_1611_9_HD
##use for 9



# -------xyzhd
N9_WGDOP_B128_0_Mcon_0_1611_9="python $train_script --ntop_candi 20-11  --cascade_step 0  --neg_loss_w 1.15-1.2-1.25 --model_epoch 32 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 1 --UseMultiConcat"
N9_WGDOP_B128_0_Mcon_1_1611_9="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 101 --UseMultiConcat"

#$N9_WGDOP_B128_0_Mcon_0_1611_9
#$N9_WGDOP_B128_0_Mcon_1_1611_9
##use for 9



N9_WGDOP_B128_0_Mcon_1_1611_9_HD_FT="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-46 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements hd --finetune --max_epoch 67 --UseMultiConcat"
#$N9_WGDOP_B128_0_Mcon_1_1611_9_HD_FT



N9_WGDOP_B128_0_Mcon_1_1611_9_XYZ_FT="python $train_script --ntop_candi 20-11  --cascade_step 1  --neg_loss_w 1.15-1.1-1.25 --model_epoch 0-46 --a N9_WGDOP_B128 --data_source data_WGDOP_new --model_config 9-52-18 --data_source_test data_WGDOP_test2 --num_pos_ls 9 --batch_size 128 --feed_star_elements xyz --finetune  --max_epoch 67 --UseMultiConcat"
#$N9_WGDOP_B128_0_Mcon_1_1611_9_XYZ_FT



N9_WGDOP_B128_0_Mcon_1_1710_FT="python $train_script --ntop_candi 17-10  --cascade_step 1  --neg_loss_w 0.75-1.1-1.25 --model_epoch 10-46 --a N9_WGDOP_B128 --data_source data_WGDOP --model_config 5-3-5  --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhd --max_epoch 97 --UseMultiConcat --finetune --log_dir logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop17-10_mc5-3-5MtiCon"

#$N9_WGDOP_B128_0_Mcon_1_1710_FT








N9_WGDOP_B128_FT="python $train_script --a N9_WGDOP_B128 --data_source data_WGDOP --num_pos_ls 9  --batch_size 128 --feed_star_elements xyzhd --finetune --model_epoch 38 --max_epoch 67"


N9_WGDOP_B256="python $train_script --a N9_WGDOP_B256 --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --max_epoch 48"
N9_WGDOP_B256_FT="python $train_script --a N9_WGDOP_B256 --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --finetune --model_epoch 36 --max_epoch 53"

N9_WGDOP_B512="python $train_script --a N9_WGDOP_B512 --data_source data_WGDOP --num_pos_ls 9 --batch_size 512 --feed_star_elements xyzhd --max_epoch 48"
N9_WGDOP_B512_FT="python $train_script --a N9_WGDOP_B512 --data_source data_WGDOP --num_pos_ls 9 --batch_size 512 --feed_star_elements xyzhd --finetune --model_epoch 36 --max_epoch 53"

#$N9_WGDOP_B128_FT


#$N9_WGDOP_B256_FT
#$N9_WGDOP_B256
#./parallel_commands "$N9_WGDOP_B48" "$N9_WGDOP_B64" "$N9_WGDOP_B128" "$N9_WGDOP_B256"

N9_WGDOP_B128_Reg="python $train_script --a N9_WGDOP_B128_Reg --data_source data_WGDOP --num_pos_ls 9 --batch_size 128 --feed_star_elements xyzhdi --max_epoch 40 --IsRegression"
#$N9_WGDOP_B128_Reg


N9_WGDOP_B256_hingloss_m98="python $train_script  --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --max_epoch 100 --IsHingeloss --model_config 98"
N9_WGDOP_B256_hingloss_m99="python $train_script  --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --max_epoch 100 --IsHingeloss --model_config 99"
N9_WGDOP_B256_hingloss_m100="python $train_script  --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --max_epoch 100 --IsHingeloss --model_config 100"
N9_WGDOP_B256_hingloss_m101="python $train_script  --data_source data_WGDOP --num_pos_ls 9 --batch_size 256 --feed_star_elements xyzhd --max_epoch 100  --model_config 101"
#$N9_WGDOP_B256_hingloss_m101
