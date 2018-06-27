#!/bin/bash

train_script=train.py
MaxEpoch=5

DataSource=data_withg


# Loss config  ***************************
LOSS0="python $train_script --a LOSS1  --empty_loss_w 0.05 --neg_loss_w 0.1  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss0"
LOSS1="python $train_script --a LOSS1  --empty_loss_w 0.05 --neg_loss_w 0.1  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss1"
LOSS2="python $train_script --a LOSS2  --empty_loss_w 0.05 --neg_loss_w 0.3  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss2"
LOSS3="python $train_script --a LOSS3  --empty_loss_w 0.05 --neg_loss_w 0.5  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss3"
LOSS4="python $train_script --a LOSS4  --empty_loss_w 0.05 --neg_loss_w 0.7  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss4"
LOSS5="python $train_script --a LOSS5  --empty_loss_w 0.05 --neg_loss_w 0.9  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss5"
LOSS6="python $train_script --a LOSS6  --empty_loss_w 0.05 --neg_loss_w 0.9  --model_config 3 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch --log_dir log_loss6"
#./parallel_commands  "$LOSS0" "$LOSS1" "$LOSS2" "$LOSS3" "$LOSS4" "$LOSS5"

# Model config  ***************************
Mc0_AllEle_B32_N11="python $train_script --a Mc0_AllEle_B32_N11  --model_config 0 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc1_AllEle_B32_N11="python $train_script --a Mc1_AllEle_B32_N11  --model_config 1 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc2_AllEle_B32_N11="python $train_script --a Mc2_AllEle_B32_N11  --model_config 2 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc3_AllEle_B32_N11="python $train_script --a Mc3_AllEle_B32_N11  --model_config 3 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc4_AllEle_B32_N11="python $train_script --a Mc4_AllEle_B32_N11  --model_config 4 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc5_AllEle_B32_N11="python $train_script --a Mc5_AllEle_B32_N11  --model_config 5 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc6_AllEle_B32_N11="python $train_script --a Mc6_AllEle_B32_N11  --model_config 6 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc7_AllEle_B32_N11="python $train_script --a Mc7_AllEle_B32_N11  --model_config 7 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc8_AllEle_B32_N11="python $train_script --a Mc8_AllEle_B32_N11  --model_config 8 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc9_AllEle_B32_N11="python $train_script --a Mc9_AllEle_B32_N11  --model_config 9 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"

#./parallel_commands "$Mc0_AllEle_B32_N11"  "$Mc1_AllEle_B32_N11" "$Mc2_AllEle_B32_N11"  "$Mc3_AllEle_B32_N11" "$Mc4_AllEle_B32_N11" "$Mc5_AllEle_B32_N11"
#./parallel_commands "$Mc6_AllEle_B32_N11" "$Mc7_AllEle_B32_N11" "$Mc8_AllEle_B32_N11" "$Mc9_AllEle_B32_N11"
#./parallel_commands  "$Mc8_AllEle_B32_N11" "$Mc9_AllEle_B32_N11"
#$Mc0_AllEle_B32_N11



Mc100_AllEle_B32_N11="python $train_script --a Mc100_AllEle_B32_N11  --model_config 100 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc101_AllEle_B32_N11="python $train_script --a Mc101_AllEle_B32_N11  --model_config 101 --data_source $DataSource"
Mc102_AllEle_B32_N11="python $train_script --a Mc102_AllEle_B32_N11  --model_config 102 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc103_AllEle_B32_N11="python $train_script --a Mc103_AllEle_B32_N11  --model_config 103 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
Mc104_AllEle_B32_N11="python $train_script --a Mc104_AllEle_B32_N11  --model_config 104 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"

#./parallel_commands "$Mc100_AllEle_B32_N11" "$Mc101_AllEle_B32_N11" "$Mc102_AllEle_B32_N11" "$Mc103_AllEle_B32_N11"
$Mc101_AllEle_B32_N11

# Eles  ***************************
AllEle_B32_N11="python $train_script --a AllEle_B32_N11 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
XyzHd_B32_N11="python $train_script --a XyzHd_B32_N11 --feed_star_elements xyzhd --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Xyz_B32_N11="python $train_script --a Xyz_B32_N11 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Xyzg_B32_N11="python $train_script --a Xyzg_B32_N11 --feed_star_elements xyzg --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Xyzn_B32_N11="python $train_script --a Xyzn_B32_N11 --feed_star_elements xyzn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Xyzgn_B32_N11="python $train_script --a Xyzgn_B32_N11 --feed_star_elements xyzgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Hd_B32_N11="python $train_script --a Hd_B32_N11 --feed_star_elements hd --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Hdg_B32_N11="python $train_script --a Hdg_B32_N11 --feed_star_elements hdg --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Hdn_B32_N11="python $train_script --a Hdn_B32_N11 --feed_star_elements hdn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
Hdgn_B32_N11="python $train_script --a Hdgn_B32_N11 --feed_star_elements hdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"

#./parallel_commands "$AllEle_B32_N11" "$XyzHd_B32_N11" "$Xyz_B32_N11" "$Xyzg_B32_N11" "$Xyzn_B32_N11" "$Xyzgn_B32_N11" "$Hd_B32_N11" "$Hdg_B32_N11" "$Hdn_B32_N11" "$Hdgn_B32_N11"


# num_loss  ***************************
AllEle_B32_N11_nl0="python $train_script --a AllEle_B32_N11_nl0 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.0 --max_epoch $MaxEpoch"
AllEle_B32_N11_nl0d01="python $train_script --a AllEle_B32_N11_nl0d01 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch $MaxEpoch"
AllEle_B32_N11_nl0d05="python $train_script --a AllEle_B32_N11_nl0d05 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.05 --max_epoch $MaxEpoch"
AllEle_B32_N11_nl0d1="python $train_script --a AllEle_B32_N11_nl0d1 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.1 --max_epoch $MaxEpoch"
AllEle_B32_N11_nl0d3="python $train_script --a AllEle_B32_N11_nl0d3 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.3 --max_epoch $MaxEpoch"
AllEle_B32_N11_nl0d8="python $train_script --a AllEle_B32_N11_nl0d8 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.8 --max_epoch $MaxEpoch"

#./parallel_commands "$AllEle_B32_N11_nl0" "$AllEle_B32_N11_nl0d01" "$AllEle_B32_N11_nl0d05" "$AllEle_B32_N11_nl0d1" "$AllEle_B32_N11_nl0d3" "$AllEle_B32_N11_nl0d8"

# AllEle  ***********************
AllEle_B32_AllNum="python $train_script --a AllEle_B32_AllNum --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
AllEle_B32_N11="python $train_script --a AllEle_B32_N11 --feed_star_elements xyzhdgn --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

AllEle_B128_AllNum="python $train_script --a AllEle_B128_AllNum --feed_star_elements xyzhdgn --batch_size 128 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
AllEle_B128_N11="python $train_script --a AllEle_B128_N11 --feed_star_elements xyzhdgn --batch_size 128 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

AllEle_B256_AllNum="python $train_script --a AllEle_B256_AllNum --feed_star_elements xyzhdgn --batch_size 256 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
AllEle_B256_N11="python $train_script --a AllEle_B256_N11 --feed_star_elements xyzhdgn --batch_size 256 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

#$AllEle_B32_AllNum

# Xyz  ***********************
Xyz_B32_AllNum="python $train_script --a Xyz_B32_AllNum --feed_star_elements xyz --batch_size 32 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Xyz_B32_N11="python $train_script --a Xyz_B32_N11 --feed_star_elements xyz --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

Xyz_B128_AllNum="python $train_script --a Xyz_B128_AllNum --feed_star_elements xyz --batch_size 128 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Xyz_B128_N11="python $train_script --a Xyz_B128_N11 --feed_star_elements xyz --batch_size 128 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

Xyz_B256_AllNum="python $train_script --a Xyz_B256_AllNum --feed_star_elements xyz --batch_size 256 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Xyz_B256_N11="python $train_script --a Xyz_B256_N11 --feed_star_elements xyz --batch_size 256 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"



# Hd  ***********************
Hd_B32_AllNum="python $train_script --a Hd_B32_AllNum --feed_star_elements hd --batch_size 32 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Hd_B32_N11="python $train_script --a Hd_B32_N11 --feed_star_elements hd --batch_size 32 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

Hd_B128_AllNum="python $train_script --a Hd_B128_AllNum --feed_star_elements hd --batch_size 128 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Hd_B128_N11="python $train_script --a Hd_B128_N11 --feed_star_elements hd --batch_size 128 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

Hd_B256_AllNum="python $train_script --a Hd_B256_AllNum --feed_star_elements hd --batch_size 256 --num_pos_ls 7-9-11-13 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"
Hd_B256_N11="python $train_script --a Hd_B256_N11 --feed_star_elements hd --batch_size 256 --num_pos_ls 11 --loss_w_class 1.0 --loss_w_numpos 0.01 --max_epoch 2"

#./parallel_commands "$AllEle_B32_AllNum" "$AllEle_B32_N11" "$AllEle_B128_AllNum" "$AllEle_B128_N11" "$AllEle_B256_AllNum" "$AllEle_B256_N11"
#./parallel_commands "$Xyz_B32_AllNum" "$Xyz_B32_N11" "$Xyz_B128_AllNum" "$Xyz_B128_N11" "$Xyz_B256_AllNum" "$Xyz_B256_N11"
#./parallel_commands "$Hd_B32_AllNum" "$Hd_B32_N11" "$Hd_B128_AllNum" "$Hd_B128_N11" "$Hd_B256_AllNum" "$Hd_B256_N11"

#;$AllEle_B32_N11
