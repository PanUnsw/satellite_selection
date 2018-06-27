#!/bin/bash

train_script=train.py
AllEle_AllNum="python $train_script --a AllEle_AllNum --feed_star_elements xyzhdgn --num_pos_ls 7-9-11-13"
Xyzhdg_AllNum="python $train_script --a Xyzhdg_AllNum --feed_star_elements xyzhdg --num_pos_ls 7-9-11-13"
Xyz_AllNum="python $train_script --a Xyz_AllNum --feed_star_elements xyz --num_pos_ls 7-9-11-13"
hd_AllNum="python $train_script --a hd_AllNum --feed_star_elements hd --num_pos_ls 7-9-11-13"


AllEle_N7="python $train_script --a AllEle_N7 --feed_star_elements xyzhdgn --num_pos_ls 7"
Xyz_N7="python $train_script --a Xyz_N7 --feed_star_elements xyz --num_pos_ls 7"
AllEle_N9="python $train_script --a AllEle_N9 --feed_star_elements xyzhdgn --num_pos_ls 9"
Xyz_N9="python $train_script --a Xyz_N9 --feed_star_elements xyz --num_pos_ls 9"
AllEle_N11="python $train_script --a AllEle_N11 --feed_star_elements xyzhdgn --num_pos_ls 11"
Xyz_N11="python $train_script --a Xyz_N11 --feed_star_elements xyz --num_pos_ls 11"
AllEle_N13="python $train_script --a AllEle_N13 --feed_star_elements xyzhdgn --num_pos_ls 13"
Xyz_N13="python $train_script --a Xyz_N13 --feed_star_elements xyz --num_pos_ls 13"

#$AllEle_N11

AllEle_N11_sync="python $train_script --a AllEle_N11 --feed_star_elements xyzhdgn --num_pos_ls 11 --data_source data_sync"
Xyz_N11_sync="python $train_script --a Xyz_N11 --feed_star_elements xyz --num_pos_ls 11 --data_source data_sync"

#./parallel_commands "$AllEle_N11_sync" "$Xyz_N11_sync" "$Xyz_N11"

#./parallel_commands "$AllEle_AllNum" "$Xyzhdg_AllNum" "$Xyz_AllNum" "$hd_AllNum"  "$AllEle_N11"
#./parallel_commands "$AllEle_N13" "$AllEle_N7" "$AllEle_N9" "$Xyz_N11" "$Xyz_N13" "$Xyz_N7" "$Xyz_N9"


AllEle_AllNum_sync_softmax="python $train_script --a AllEle_AllNum_Sync_SoftMaxAc --feed_star_elements xyzhdgn --num_pos_ls 7-9-11-13 --data_source data_sync "
AllEle_AllNum_sync_m6="python $train_script --a AllEle_AllNum_Sync_m6 --feed_star_elements xyzhdgn --num_pos_ls 7-9-11-13 --data_source data_sync --model_config 6"
Xyzhdg_AllNum_sync="python $train_script --a AllEle_AllNum_Sync --feed_star_elements xyzhdg --num_pos_ls 7-9-11-13 --data_source data_sync"
#$AllEle_AllNum_sync_softmax


# pos num los
NpnL_AllEle_AllNum_sync="python $train_script --a NpnL_AllEle_AllNum_Sync --feed_star_elements xyzhdgn --num_pos_ls 7-9-11-13 --data_source data_sync --loss_w_numpos 0.0"
NpnL_AllEle_N11_sync="python $train_script --a NpnL_AllEle_N11_sync --feed_star_elements xyzhdgn --num_pos_ls 11 --data_source data_sync --loss_w_numpos 0.0"
AllEle_AllNum_sync_npnl="python $train_script --a AllEle_AllNum_Sync_npnl --feed_star_elements xyzhdgn --num_pos_ls 7-9-11-13 --data_source data_sync --loss_w_numpos 0.6"
#./parallel_commands "$AllEle_AllNum_sync_npnl" "$AllEle_AllNum_sync_npnl"
#./parallel_commands "$NpnL_AllEle_AllNum_sync" "$NpnL_AllEle_N11_sync"




FineTune_AllEle_N11="python $train_script --log_dir LOG_AllEle_N11_xyzhdgn_b32_11_mc101  --finetune --loss_w_numpos 0.0"
FineTune_AllEle_N11_nl="python $train_script --log_dir LOG_AllEle_N11_xyzhdgn_b32_11_mc101_nl  --finetune --loss_w_numpos 0.5"
#$FineTune_AllEle_N11
#$FineTune_AllEle_N11_nl


# withg data
N11_withg="python $train_script --a N11_withg --data_source data_withg"
AllNum_withg="python $train_script --a AllNum_withg --data_source data_withg --num_pos_ls 7-9-11-13"
#./parallel_commands "$N11_withg" "$AllNum_withg"

N9_WGDOP="python $train_script --a N9_WGDOP_TMP --data_source data_WGDOP --num_pos_ls 9"
N9_WGDOP_ECL="python $train_script --a N9_WGDOP_ECL --data_source data_WGDOP --num_pos_ls 9 --UseErrCondLabel"
N11="python $train_script --a N11"
N9_WDOP_FT="python $train_script --a N9_WDOP --data_source data_WPDOP --num_pos_ls 9 --finetune --model_epoch 3 --max_epoch 10"
#$N9_WGDOP_ECL
$N9_WGDOP
