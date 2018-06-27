# xyz Nov 2017

from __future__ import print_function
import pdb, traceback
import numpy as np
import os
import sys
import glob
import time
from numpy.linalg import inv
import matplotlib.pyplot as plt
from io import StringIO
import math

file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_new_ntop20-11_mc9-52-18MtiCon/eval_pred_clean_fixpn.txt"
#file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-11_mc9-51-8MtiCon/eval_pred_clean_fixpn.txt"
#file_name = "/home/y/star_selection/RES/GOOD-logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-11_mc9-51-8MtiCon/eval_pred_clean_fixpn.txt" #maxpooling 3conv(use)
#file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-11_mc9-50-8MtiCon/eval_pred_clean_fixpn.txt" #maxpooling 4 conv
#file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_ntop16-11_mc9-5-8MtiCon/eval_pred_clean_fixpn.txt" # conv compare (not use any more)
file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_12_ntop20-13_mc9-53-18MtiCon/eval_pred_clean_fixpn.txt" # maxpooling use
#file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_12_data_WGDOP_new2_ntop20-13_mc9-50-18MtiCon/eval_pred_clean_fixpn.txt" # maxpooling use

file_name = "/home/y/star_selection/RES/logN9_WGDOP_B128_xyzhd_b128_9_data_WGDOP_9_ntop20-11_mc9-53-18MtiCon/eval_pred_clean_fixpn.txt"

raw_data = np.loadtxt(file_name).astype(np.float64)
k=0

num_sel = 20
#num_sel = 20

num = raw_data.shape[0]/num_sel

red_rate = np.zeros((num,1))
DOP_TRU = np.zeros((num,1))
DOP_evl = np.zeros((num,1))
for i in range(0,num):
    Obs_one = raw_data[i*num_sel:(i*num_sel)+num_sel, 1:4]
    Q = raw_data[i*num_sel:(i*num_sel)+num_sel, 4]
    evl_idx = np.nonzero(raw_data[i*num_sel:(i*num_sel)+num_sel, 8])[0]
    tru_idx = np.nonzero(raw_data[i * num_sel:(i * num_sel) + num_sel, 9])[0]

    H_s = (evl_idx.shape[0], 4)
    H_evl = np.ones(H_s)
    H_tru = np.ones(H_s)

    H_evl[:,0:3] = Obs_one[evl_idx,:]

    H_tru[:,0:3] = Obs_one[tru_idx,:]


    invQ = np.power(np.sin(Q),2 )

##############3
    #invQ = np.sin(Q)

    invQ_evl = invQ[evl_idx]
    #H_evl[:, 3] = invQ_evl
##############
    H_H_evl = np.dot(np.transpose(H_evl),  np.diag(invQ_evl) )
    H_H_evl = np.dot(H_H_evl, H_evl)


    #H_H_evl = np.dot(np.transpose(H_evl), H_evl)



    H_H_evl_in = inv(H_H_evl)
    H_H_evl_in_tr = np.sqrt(np.trace(H_H_evl_in))




    ##############3
    invQ_tru = invQ[tru_idx]

    #H_tru[:, 3] = invQ_tru
    ##############

    H_H_tru = np.dot(np.transpose(H_tru),  np.diag(invQ_tru))
    H_H_tru = np.dot(H_H_tru, H_tru)

    #H_H_tru = np.dot(np.transpose(H_tru), H_tru)

    H_H_tru_in = inv(H_H_tru)

    H_H_tru_in_tr = np.sqrt(np.trace(H_H_tru_in))

    red_rate[k, 0] = np.divide(np.subtract(H_H_evl_in_tr,H_H_tru_in_tr),H_H_tru_in_tr)
    DOP_TRU[k, 0] = H_H_tru_in_tr
    DOP_evl[k, 0] = H_H_evl_in_tr

    k+=1

idx = red_rate[np.where( red_rate <0.001 ) ]
prec = (idx.shape[0])/float(red_rate.shape[0])


print(prec)


DOP_DIFF = DOP_evl - DOP_TRU
idx = DOP_DIFF[np.where( DOP_DIFF <0.01 ) ]
idx_s_z = np.where( red_rate >0.3 )


prec = (idx.shape[0])/float(DOP_DIFF.shape[0])

print(prec)



print( (np.mean(DOP_TRU)))
print( (np.mean(DOP_evl)))

pre_in=( (np.mean(DOP_evl))- (np.mean(DOP_TRU))) / (np.mean(DOP_TRU))
print(pre_in)
#plt.plot(idx)
#plt.show()
idx_dop_tr = np.where( DOP_TRU <4.4)[0]
idx_pre = np.where( red_rate >0.2 )[0]


num_equ = []

for x in idx_pre:
    if x in idx_dop_tr:
        num_equ.append(x)


plt.plot(DOP_TRU)
plt.show()

plt.plot(DOP_DIFF)
plt.show()


plt.plot(DOP_evl)
plt.show()
plt.plot(red_rate)
plt.show()


