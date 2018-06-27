# End to end satellite selection with deep learning

## (1)
The codes are tested under Ubuntu16.04 & Ubuntu17.10, python2, and Tensorflow1.6 & TF1.8 (GPU).
## (2)
Download data from: (326M)
  https://unsw-my.sharepoint.com/:u:/g/personal/z5105843_ad_unsw_edu_au/EauUY9zffq1LpEW1OtTzxyUBed34PAB4XqoIBWHBWMACnA?e=13eGQY
Unzip to root directory ("satellite_selection"). Then four data folders will under "satellite_selection/": 
	- data_WGDOP_new(366M)
	- data_WGDOP_new2(311M) 
	- data_WGDOP_test2(63M) 
	- data_WGDOP_test_new2(31M)
## (3)
train with script: run_train.sh
evaluate with script: run_evaluate.sh
The results will be saved in RES/ and log name is automatically set by training parameters.
