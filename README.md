# End to end satellite selection with deep learning

(1) The codes are tested under Ubuntu16.04 & Ubuntu17.10, python2, and Tensorflow1.6 & TF1.8 (GPU).  
>
(2) Download test data from:  https://drive.google.com/open?id=1LYO8Vg2rVoRC0OqtfDs4_jxw74OkVpc7    
  Unzip to root directory ("satellite_selection"). Then two data folders will under "satellite_selection/":       
 * data_WGDOP_test2(63M)    
 * data_WGDOP_test_new2(31M)    
>

(3) Download trained model from: https://drive.google.com/open?id=1jp1jJuMfsOZRUjBxd90QyGyS420DIeV5    
  Unzip to root dir: satellite/RES  
(4) train with script: run_train.sh   
evaluate with script: run_evaluate.sh    
Some parameters may have to be modified in run_evaluate.sh   
The results will be saved in RES/ and log name is automatically set by training parameters.  
