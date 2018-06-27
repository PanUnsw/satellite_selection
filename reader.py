# xyz Nov 2017

from __future__ import print_function
import pdb, traceback
import numpy as np
import os
import sys
import glob
import time
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

t0 = time.time()

def get_file_list(data_dir):
    file_name_ls = []
    folder_list = os.listdir(data_dir)
    for folder in folder_list:
        folder = os.path.join( data_dir,folder )
        file_name_ls += glob.glob( os.path.join(folder,'*.txt') )
    return file_name_ls


def File_Transfer(default_file_name_ls):
    for file_name in default_file_name_ls:
        data = np.loadtxt(file_name).astype(np.float)
        new_file_name = os.path.splitext(file_name)[0]
        np.save(new_file_name,data)
        print('%s -> %s'%(os.path.basename(file_name),os.path.basename(new_file_name)+'.npy'))

class Star_Reader():
    '''
    self.raw_ele_idxs: [ txyzgihdln ] the element index in the raw input file, added with pos num as the end
    self.candidate_feed_elements: [ xyzhdgn ]
    '''
    max_num_instars = 56
    data_summary_str = ''

    def __init__(self,file_name_ls = None,train_rate=0.7,data_source='data_sync',raw_elements = 'txyzgihdl',
                 IsOnlyEval=False,IsUseErrCondLabel=False,IsEmptyLabel=False,IsRegression=False ):

        self.raw_elements = raw_elements
            # g:star group(1,2,3); i: star idx(1~56); l:ground truth label
        self.raw_ele_idxs = {}
        for i,e in enumerate(self.raw_elements):
            self.raw_ele_idxs[e] = i
        self.raw_ele_idxs['n'] = len(raw_elements)
        self.num_raw_ele = len(self.raw_elements)
        self.empty_vec =np.array([[[0]*self.num_raw_ele]])

        ## the elements of each column in raw fiile
        #if 'p' not in raw_elements:
        #    self.candidate_feed_elements = 'xyzhdgin'  # the real feed_star_elements is a sub set of this
        #else:
        #    self.candidate_feed_elements = 'xyzhdginp'
        self.candidate_feed_elements = 'xyzhdgin'

        self.candi_feed_ele_idxs = {}
        for i, e in enumerate(self.candidate_feed_elements):
            self.candi_feed_ele_idxs[e] = i


        if file_name_ls == None:
            self.DATA_DIR =DATA_DIR= os.path.join(BASE_DIR,data_source)
            self.DEFAULT_ZIP_DATA_PATH = DATA_DIR+'/datas_labels'
            self.DEFAULT_ZIP_DATA_SHUFFLED_PATH = self.DEFAULT_ZIP_DATA_PATH + '_shuffled'
            file_name_ls = get_file_list(DATA_DIR)

        if self.load_saved_data() == False:
            self.load_file_ls(file_name_ls)
            self.save_shufled()

        if IsUseErrCondLabel:
            self.LoadErrCondLabel()

        self.gen_train_test(train_rate,IsEmptyLabel,IsRegression)
        #self.add_all_data_rates_str()
        print('read data init t=%0.3f'%(time.time()-t0))

    def LoadErrCondLabel(self):
        path = os.path.join(self.DATA_DIR,'label_errcondition.npy')
        assert os.path.exists(path), 'find no: %s'%(path)
        err_cond_label = np.load(path)
        self.labels = err_cond_label
        print('load %s'%(path))
        print('label_errcondition.npy shape:',self.labels.shape)

    def get_train_test(self,feed_star_elements='xyzhdgn',num_pos_ls=None):
        if num_pos_ls == None:
            train_test_data_label = self.train_test_data_label[0:4] # [train_data,train_label, test_data,test_label,shuffled_idx]
        else:
            train_test_data_label,data_rate_str = self.select_num_pos(num_pos_ls)
            #self.data_summary_str = data_rate_str

        feed_ele_idx_in_candi = [self.candi_feed_ele_idxs[e] for e in feed_star_elements]
        # select feeding elements
        train_test_data_label[0] = train_test_data_label[0][:,:,feed_ele_idx_in_candi]
        train_test_data_label[2] = train_test_data_label[2][:,:,feed_ele_idx_in_candi]
        return train_test_data_label

    def load_saved_data(self):
        if self.load_saved_data_path(self.DEFAULT_ZIP_DATA_SHUFFLED_PATH +'.npz'):
            print('load shuffled: %s'%(self.DEFAULT_ZIP_DATA_SHUFFLED_PATH +'.npz'))
            return True
        else:
            if self.load_saved_data_path(self.DEFAULT_ZIP_DATA_PATH +'.npz'):
                # save shufled
                self.save_shufled()
                print('load %s \n'%(self.DEFAULT_ZIP_DATA_PATH +'.npz'))
                return True
            else:
                return False
    def save_shufled(self):
        if not os.path.exists(self.DEFAULT_ZIP_DATA_SHUFFLED_PATH +'.npz'):
            N = self.datas.shape[0]
            shuffled_idx = np.arange(N)
            np.random.shuffle(shuffled_idx)
            self.datas = self.datas[shuffled_idx, ...]
            self.labels = self.labels[shuffled_idx, ...]
            self.save_train_test(self.DEFAULT_ZIP_DATA_SHUFFLED_PATH)
            print('save shuffled: %s' % (self.DEFAULT_ZIP_DATA_SHUFFLED_PATH + '.npz'))

    def load_saved_data_backup(self):
        return self.load_saved_data_path(self.DEFAULT_ZIP_DATA_PATH + '.npz')

    def load_saved_data_path(self,saved_data_path):
        if os.path.exists(saved_data_path):
            D = np.load(saved_data_path)
            self.datas,self.labels = [D['datas'], D['labels']]
            self.data_summary_str = np.array_str(D['data_summary_str'])
            print('load data from %s   t=%0.3f'%(saved_data_path,time.time()-t0))
            print('datas shape: ',self.datas.shape)
            return True
        else:
            print('no npz data, reading from txts')
            return False

    def load_file_ls(self,file_name_ls):
        print('start reading %d files:'%len(file_name_ls))
        org_data_pn_ls = []
        t_last_file = -10
        for n,file_name in enumerate(file_name_ls):
            print('file %d    t= %0.3f'%(n,time.time()-t0))
            org_data_pn_i = self.load_file(file_name,t_last_file)
            org_data_pn_ls.append(org_data_pn_i)
            t_last_file = org_data_pn_i[-1,-1,self.raw_ele_idxs['t']]
        org_data_pn = np.concatenate(org_data_pn_ls,axis=0)

        # [N,max_num_instars,num_candi_feed_ele]
        candidate_feed_elements_idxs_inraw = [self.raw_ele_idxs[e] for e in self.candidate_feed_elements]
        self.datas = org_data_pn[:,:,candidate_feed_elements_idxs_inraw].astype('float32')
        # [N,max_num_instars]
        self.labels = org_data_pn[:,:,self.raw_ele_idxs['l']].astype('int32')

        self.data_summary_str = 'file num: %d \ntotal time steps: %d \n'%(
            len(file_name_ls),org_data_pn.shape[0])
        print(self.data_summary_str)

    def load_file(self,file_name,t_last_file=0):
        '''
            t_last_file is the end t of last file.
            Add (t_last_file+10) to all t of current file, which indicates that
            the time steps between each file are not continuous.
        '''
        t_last_file += 10
        if os.path.splitext(file_name)[1]=='.txt':
            raw_data = np.loadtxt(file_name).astype(np.float32)
        if os.path.splitext(file_name)[1]=='.npy':
            raw_data = np.load(file_name).astype(np.float32)

        data_ls_ls = []
        t0 = raw_data[0,0]
        # read all the satelites vec of same time step to the same sub_list
        for i in range(0,raw_data.shape[0]):
            line = raw_data[i,:]
            t = int(line[0]-t0) # use t (from 0) as the list of list index
            while len(data_ls_ls)<t+1:
                # new time step
                data_ls_ls.append([])
            star_vec_i = line
            star_vec_i[0] += t_last_file
            star_vec_i = np.reshape(star_vec_i,[1,1,-1])
            data_ls_ls[t].append(star_vec_i)
            # data_ls_ls[t] is all the data for timestep t

        # delete empty time steps
        data_ls_ls = [ls for ls in data_ls_ls if ls]

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
        org_data = np.concatenate(data_ls,axis=0).astype(np.float32)
        # [N,max_num_instars,num_raw_ele]
        org_data = self.sort_staridx(org_data)
            # [N,max_num_instars,num_raw_ele+1]
        org_data_pn = self.add_num_pos(org_data)
        return org_data_pn

    def add_num_pos(self,org_data):
        '''
        input: org_data # [N,max_num_instars,num_raw_ele]
                elements = xyzgil
        output: org_data_pn # [N,max_num_instars,num_raw_ele+1]
                elements = xyzgiln
        '''
        org_data_pn = np.lib.pad(org_data,((0,0),(0,0),(0,1)),'constant', constant_values=(0,0))
        for i in range(org_data_pn.shape[0]):
            num_pos = np.sum( org_data_pn[i,:,self.raw_ele_idxs['l']],axis=-1 )
            for j in range(org_data_pn.shape[1]):
                if org_data_pn[i,j,self.raw_ele_idxs['x']]!=0:
                    org_data_pn[i,j,self.raw_ele_idxs['n']]=num_pos
                else:
                    org_data_pn[i,j,self.raw_ele_idxs['n']]=0

        return org_data_pn


    def sort_staridx(self,org_data):
        '''
        # [N,max_num_instars,num_raw_ele]
        sort by star_idx org_data[:,:,-2]
        '''
        sorted_data = np.zeros(shape=org_data.shape)

      # # idx = np.argsort(org_data[:,:,-2])
      # # for i in range(org_data.shape[0]):
      # #     sorted_data[i,:] = org_data[i,:][idx[i]]
        for i in range(org_data.shape[0]):
            for j in range(org_data.shape[1]):
                star_idx = int(org_data[i,j,self.raw_ele_idxs['i']])
                if star_idx > 0:
                    sorted_data[i,star_idx-1,:] = org_data[i,j,:]
        return sorted_data

    @staticmethod
    def get_empty_idx(datas):
        empty_idx = np.zeros(shape=(datas.shape[0],datas.shape[1]))
        for n in range(datas.shape[0]):
            for m in range(datas.shape[1]):
                if np.sum(datas[n,m,0:3]) == 0:
                    empty_idx[n,m] = 1
        return empty_idx

    @staticmethod
    def AddEmptyLabel(datas,labels):
        empty_idx = Star_Reader.get_empty_idx(datas)
        for n in range(datas.shape[0]):
            for m in range(datas.shape[1]):
                if empty_idx[n,m]==1:
                    labels[n,m] = 2
        return labels

    def gen_train_test(self,train_rate,IsEmptyLabel,IsRegression):
        if IsEmptyLabel:
            self.labels = Star_Reader.AddEmptyLabel(self.datas,self.labels)
        if not os.path.exists(self.DEFAULT_ZIP_DATA_PATH + '.npz'):
            self.save_train_test(self.DEFAULT_ZIP_DATA_PATH)

        N = self.datas.shape[0]
        if IsRegression:
            shuffled_label = self.get_pos_satelite_idx(self.datas,self.labels)
        else:
            shuffled_label = self.labels

        h_d_idxs = [self.candi_feed_ele_idxs['h'],self.candi_feed_ele_idxs['d']]
        self.datas[:,:,h_d_idxs] = self.datas[:,:,h_d_idxs] * math.pi / 180.0

        h_idxs = [self.candi_feed_ele_idxs['h']]
        #self.datas[:, :, h_idxs] = np.power(np.sin(self.datas[:, :, h_idxs]),2)


        q = (np.sin(self.datas[:,:,h_idxs]))

        q = np.tile(q,(1,1,3))
        xyz_idxs = [self.candi_feed_ele_idxs['x'], self.candi_feed_ele_idxs['y'], self.candi_feed_ele_idxs['z']]
        #self.datas[:, :, xyz_idxs] = self.datas[:, :, xyz_idxs] * q


        shuffled_data = self.datas

        train_N = int(N*train_rate)
        train_data = shuffled_data[0:train_N,...]
        test_data = shuffled_data[train_N+1:N,...]
        train_label = shuffled_label[0:train_N,...]
        test_label = shuffled_label[train_N+1:N,...]
        self.data_summary_str += 'train num: %d \ntest num: %d\n'%(train_data.shape[0],test_data.shape[0])
        self.train_test_data_label = [train_data,train_label, test_data,test_label]

    def get_pos_satelite_idx(self,datas,labels):
        num_pos = np.max(datas[0,:,self.candi_feed_ele_idxs['n']]).astype(int)
        pos_idxs = np.zeros((labels.shape[0],num_pos))
        for i in range(labels.shape[0]):
            k = 0
            for j in range(labels.shape[1]):
                if labels[i,j] == 1:
                    pos_idxs[i,k] = datas[i,j,self.candi_feed_ele_idxs['i']]
                    k += 1
        return pos_idxs


    def save_train_test(self,filenmae):
        np.savez_compressed(filenmae,datas=self.datas,labels=self.labels,
                            data_summary_str=self.data_summary_str)
        print('train and test data saved as %s'%(self.DEFAULT_ZIP_DATA_PATH))



#    def pre_chang_group_to_two(self,feed_star_elements):
#        self.candidate_feed_elements += 'G'
#        self.candi_feed_ele_idxs['G'] = len(self.candidate_feed_elements)-1
#        feed_star_elements = feed_star_elements.replace("g","gG")
#        return feed_star_elements
#
  #  def chang_group_to_two(self,data):
  #      # before selecting elements
  #      shape = data.shape
  #      shape_new = shape[0:2]+(shape[2]+1,)
  #      data_new = np.zeros(shape=shape_new)
  #      data_new[:,:,0:self.candi_feed_ele_idxs['g']] = data[:,:,0:self.candi_feed_ele_idxs['g']]
  #      data_new[:,:,self.candi_feed_ele_idxs['n']] = data[:,:,-1]
  #      data_new[:,:,self.candi_feed_ele_idxs['g']] = data[:,:,self.candi_feed_ele_idxs['g']]
  #      for i in range(shape[0]):
  #          for j in range(shape[1]):
  #              data_new[i,j,0:-1] = data[i,j,:]
  #              g = data[i,j,self.candi_feed_ele_idxs['g']]
  #              if g == 0:
  #                  data_new[i,j,self.candi_feed_ele_idxs['g']] = 0
  #                  data_new[i,j,self.candi_feed_ele_idxs['G']] = 0
  #              if g == 1:
  #                  data_new[i,j,self.candi_feed_ele_idxs['g']] = 1
  #                  data_new[i,j,self.candi_feed_ele_idxs['G']] = 0
  #              if g == 2:
  #                  data_new[i,j,self.candi_feed_ele_idxs['g']] = 1
  #                  data_new[i,j,self.candi_feed_ele_idxs['G']] = 1
  #      return data_new


    def add_all_data_rates_str(self):
        num_pos_list = range(7,18)
        for num_pos in num_pos_list:
            _,data_rate_str = self.select_num_pos([num_pos])
            self.data_summary_str += data_rate_str

    def select_num_pos(self,num_pos_ls):
        train_data,train_label, test_data,test_label = self.train_test_data_label

        def get_idx(data,num_pos_ls):
            train_idx = []
            for i in range(data.shape[0]):
                include = False
                for num_pos in num_pos_ls:
                    if (data[i,:,self.candi_feed_ele_idxs['n']] == num_pos).any():
                        include = True
                if include:
                    train_idx.append(i)
            return train_idx

        train_idxs = get_idx(train_data,num_pos_ls)
        test_idxs = get_idx(test_data,num_pos_ls)
        new_train_data = train_data[train_idxs,:,:]
        new_train_label = train_label[train_idxs,:]
        new_test_data = test_data[test_idxs,:,:]
        new_test_label = test_label[test_idxs,:]
        if train_data.shape[0]==0:
            train_data_rate = 0
        else:
            train_data_rate = 1.0*new_train_data.shape[0]/train_data.shape[0]
        if test_data.shape[0] == 0:
            test_data_rate = 0
        else:
            test_data_rate = 1.0*new_test_data.shape[0]/test_data.shape[0]

        if train_data_rate>0 or test_data_rate>0:
            data_rate_str = 'num_pos:%s \t train rate:%0.2f \t test rate: %0.2f\n'%(
                str(num_pos_ls),train_data_rate,test_data_rate)
        else:
            data_rate_str = ''

       # print('num_pos_ls: %s'%(str(num_pos_ls)))
       # print('train num  %d -> %d   %0.3f'%(train_data.shape[0],new_train_data.shape[0],
       #                                      1.0*new_train_data.shape[0]/train_data.shape[0]))
       # print('test num = %d -> %d  %0.3f'%(test_data.shape[0],new_test_data.shape[0],
       #                        1.0*new_test_data.shape[0]/test_data.shape[0]))
        return [new_train_data,new_train_label,new_test_data,new_test_label], data_rate_str


    def gen_seq_data(self,org_data_pn,seq_num_steps,seq_batch_size,stride=None):
        '''
        from [num_timesteps,num_star,num_raw_ele] to
             [num_batch,seq_batch_size,num_step,num_star,num_raw_ele]
        '''
        num_star = org_data_pn.shape[1]
        num_star_ele = org_data_pn.shape[2]
        num_timesteps = org_data_pn.shape[0]
        last_n = 0
        seq_data_ls = []
        if stride == None:
            stride = int(seq_num_steps/2)
        while (True):
            end_n = last_n + seq_num_steps*seq_batch_size
            if end_n > num_timesteps:
                break
            seq_data_i = org_data_pn[last_n:end_n,:,:]
            seq_data_i = np.reshape(
                seq_data_i,[1,seq_batch_size,seq_num_steps,num_star,num_star_ele])
            seq_data_ls.append(seq_data_i)
            last_n += stride
        seq_data = np.concatenate(seq_data_ls,axis=0)
        #print('num_batch:%d'%(seq_data.shape[0]))
        return seq_data

    def get_valid_sat_num(self):
        empty_idx = self.get_empty_idx(self.datas)
        empty_num = np.sum(empty_idx,axis=1)
        empty_num_max_mean_min = np.array([ np.max(empty_num), np.mean(empty_num),np.min(empty_num)])
        valid_num_min_mean_max = self.max_num_instars - empty_num_max_mean_min
        print('valid min, mean, max num: ',valid_num_min_mean_max )


def main():
    num_pos_ls = [9]
    num_pos_ls=None
    feed_star_elements = 'xyzhd'
    data_source = 'data_WGDOP'
    raw_elements = 'txyzgihdlp'

    star_reader = Star_Reader(data_source=data_source,raw_elements=raw_elements)
    #star_reader.get_valid_sat_num()
    #star_reader = Star_Reader(data_source=data_source)
    train_data,train_label, test_data,test_label = star_reader.get_train_test(
        feed_star_elements=feed_star_elements,num_pos_ls=num_pos_ls)
    print(star_reader.data_summary_str)
    import pdb; pdb.set_trace()  # XXX BREAKPOINT

if __name__ == '__main__':
    try:
        main()
    except:
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
