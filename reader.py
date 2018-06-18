# xyz Nov 2017

import numpy as np
import os
import sys
import glob
from sets import Set

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR,'data')

default_file_name_ls = glob.glob( os.path.join(DATA_DIR,'d*.txt') )

class Star_Reader():
    seq_num_steps = 10
    seq_batch_size = 16
    max_num_instars = 56
    input_channel = 4   # [ x,y,z,group ]
    empty_vec =np.array([[[0,0,0,0,0,0]]])

    data_summary_str = ''

    def __init__(self,file_name_ls = default_file_name_ls ):
        self.load_file_ls(file_name_ls)

    def load_file(self,file_name):
        raw_data = np.loadtxt(file_name)
        data_ls_ls = []
        t0 = raw_data[0,0]
        for i in range(0,raw_data.shape[0]):
            line = raw_data[i,:]
            t = int(line[0]-t0)
            while len(data_ls_ls)<t+1:
                data_ls_ls.append([])
            star_vec_i = line[1:]
            star_vec_i = np.reshape(star_vec_i,[1,1,-1])
            data_ls_ls[t].append(star_vec_i)
            # data_ls_ls[t] is all the data for timestep t
        data_ls = []
        num_star_ls = []
        for ls in data_ls_ls:
            num_star_ls.append(len(ls))
            if len(ls) == 0:
                continue
            if (np.array(ls)==0).all():
                continue
            if len(ls)>=self.max_num_instars:
                ls = ls[0:self.max_num_instars]
                print('cut data')
            while len(ls) < self.max_num_instars:
                ls.append(self.empty_vec)
            data_ls.append(np.concatenate(ls,axis=1))
        org_data = np.concatenate(data_ls,axis=0)   # [N,max_num_instars,num_channel]
        org_data = self.sort_staridx(org_data)
        #[num_batch,seq_batch_size,num_step,num_star,num_channel]
        seq_data = self.gen_seq_data(org_data,self.seq_num_steps,self.seq_batch_size)
        return org_data,seq_data

    def load_file_ls(self,file_name_ls):
        org_data_ls = []
        seq_data_ls = []
        for file_name in file_name_ls:
            org_data_i,seq_data_i = self.load_file(file_name)
            org_data_ls.append(org_data_i)
            seq_data_ls.append(seq_data_i)
        org_data = np.concatenate(org_data_ls,axis=0)
        seq_data = np.concatenate(seq_data_ls,axis=0)

        self.input_datas = {}
        self.gt_datas = {}
        self.input_datas['org'] = org_data[:,:,0:self.input_channel].astype('float32')
        self.gt_datas['org'] = org_data[:,:,-1].astype('int32')

        self.input_datas['seq'] = seq_data[:,:,:,:,0:self.input_channel].astype('float32')
        self.gt_datas['seq'] = seq_data[:,:,:,:,-1].astype('int32')

        self.data_summary_str = 'file num: %d \ntotal time steps: %d '%(
            len(file_name_ls),org_data.shape[0])
        print(self.data_summary_str)

    def sort_staridx(self,org_data):
        '''
        # [N,max_num_instars,num_channel]
        sort by star_idx org_data[:,:,-2]
        '''
        sorted_data = np.zeros(shape=org_data.shape)

       # idx = np.argsort(org_data[:,:,-2])
       # for i in range(org_data.shape[0]):
       #     sorted_data[i,:] = org_data[i,:][idx[i]]
        for i in range(org_data.shape[0]):
            for j in range(org_data.shape[1]):
                star_idx = int(org_data[i,j,-2])
                if star_idx > 0:
                    sorted_data[i,star_idx-1,:] = org_data[i,j,:]
        return sorted_data

    def get_seq_input(self):
        return self.input_datas['seq']  # [num_batch,seq_batch_size,num_step,num_star,input_channel]
    def get_seq_gt(self):
        return self.gt_datas['seq'] # [num_batch,seq_batch_size,num_step,num_star]

    def get_org_input(self):
        return self.input_datas['org']  # [num_timesteps,num_star,input_channel]
    def get_org_gt(self):
        return self.gt_datas['org'] # [num_timesteps,num_star]

    def get_train_test(self,data_type,train_rate=0.7):
        N = self.input_datas[data_type].shape[0]
        train_N = int(N*train_rate)
        train_data = self.input_datas[data_type][0:train_N,...]
        test_data = self.input_datas[data_type][train_N+1:N,...]
        train_label = self.gt_datas[data_type][0:train_N,...]
        test_label = self.gt_datas[data_type][train_N+1:N,...]
        self.data_summary_str += 'train num: %d \ntest num: %d'%(train_data.shape[0],test_data.shape[0])
        return train_data,train_label, test_data,test_label


    def gen_seq_data(self,org_data,seq_num_steps,seq_batch_size,stride=None):
        '''
        from [num_timesteps,num_star,num_channel] to
             [num_batch,seq_batch_size,num_step,num_star,num_channel]
        '''
        num_star = org_data.shape[1]
        num_channel = org_data.shape[2]
        num_timesteps = org_data.shape[0]
        last_n = 0
        seq_data_ls = []
        if stride == None:
            stride = int(seq_num_steps/2)
        while (True):
            end_n = last_n + seq_num_steps*seq_batch_size
            if end_n > num_timesteps:
                break
            seq_data_i = org_data[last_n:end_n,:,:]
            seq_data_i = np.reshape(seq_data_i,[1,seq_batch_size,seq_num_steps,num_star,num_channel])
            seq_data_ls.append(seq_data_i)
            last_n += stride
        seq_data = np.concatenate(seq_data_ls,axis=0)
        #print('num_batch:%d'%(seq_data.shape[0]))
        return seq_data


    def load_fl_old(self,file_name_ls = default_file_name_ls ):
        # [ time_step, x,y,z, star_group_idx, star_idx, label ]
        raw_data_ls = []
        t_last = 0
        for file_name in file_name_ls:
            # make the timesteps in all files continuous
            raw_data_i = np.loadtxt(file_name)
            t_min = np.min(raw_data_i[:,0])
            raw_data_i[:,0] += (t_last - t_min + 1)
            t_last = np.max(raw_data_i[:,0])
            raw_data_ls.append( raw_data_i )
        raw_data = np.concatenate(raw_data_ls,axis=0).astype('float32')
        print('total %d timesteps'%(raw_data.shape[0]))

        data_ls_ls = []
        t0 = raw_data[0,0]
        for i in range(0,raw_data.shape[0]):
            line = raw_data[i,:]
            t = int(line[0]-t0)
            while len(data_ls_ls)<t+1:
                data_ls_ls.append([])
            star_vec_i = line[1:]
            star_vec_i = np.reshape(star_vec_i,[1,1,-1])
            data_ls_ls[t].append(star_vec_i)
            # data_ls_ls[t] is all the data for timestep t
        data_ls = []
        num_star_ls = []
        for ls in data_ls_ls:
            num_star_ls.append(len(ls))
            if len(ls) == 0:
                continue
            if len(ls)>=self.max_num_instars:
                ls = ls[0:self.max_num_instars]
                print('cut data')
            while len(ls) < self.max_num_instars:
                ls.append(self.empty_vec)
            data_ls.append(np.concatenate(ls,axis=1))
        data = np.concatenate(data_ls,axis=0)

        num_star = np.array(num_star_ls)
        print('empty timesteps: %d'%(np.sum(num_star==0) ))
        print('real max star num: %d'%(np.max(num_star_ls)))
        print('set max star num: %d'%(self.max_num_instars))

        self.input_data = data[:,:,0:self.input_channel].astype('float32')
        self.gt_data = data[:,:,-1].astype('int32')


if __name__ == '__main__':
    star_reader = Star_Reader()
    train_data,train_label, test_data,test_label = star_reader.get_train_test('org')
    train_data_,train_label_, test_data_,test_label_ = star_reader.get_train_test('seq')

