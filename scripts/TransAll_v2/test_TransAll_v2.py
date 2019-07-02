import numpy as np
import tensorflow as tf
import os
import time
from optparse import OptionParser

import sys
sys.path.append('./')
from utils.trainset import TrainSet
from utils.process_config import *
from TransX.TransAll_v2 import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class TestSet(object):
    # init testset by its dirctory path
    def __init__(self, dir_name):
        # load entity_id
        self.entity2id = {}
        self.id2entity = {}
        with open(os.path.join(dir_name, 'entity2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.entity2id[line_arr[0]] = int(line_arr[1])
                self.id2entity[int(line_arr[1])] = line_arr[0]
        # load relation_id
        self.relation2id = {}
        self.id2relation = {}
        with open(os.path.join(dir_name, 'relation2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.relation2id[line_arr[0]] = int(line_arr[1])
                self.id2relation[int(line_arr[1])] = line_arr[0]
        
        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        print 'dataset {} has {} entities.'.format(dir_name.split('/')[-1], self.entity_size)
        print 'dataset {} has {} relations.'.format(dir_name.split('/')[-1], self.relation_size)
        # load trip
        self.fb_h = []
        self.fb_r = []
        self.fb_l = []
        self.ok = set()
        self._load(os.path.join(dir_name, 'test.txt'), True)
        self._load(os.path.join(dir_name, 'train.txt'), False)
        self._load(os.path.join(dir_name, 'valid.txt'), False)
    
    # load function: load trip from file
    def _load(self, filename, flag):
        with open(filename) as fr:
            for line in fr:
                line_arr = line.strip().split()
                tmp_head = self.entity2id[line_arr[0]]
                tmp_last = self.entity2id[line_arr[1]]
                tmp_relation = self.relation2id[line_arr[2]]
                if flag:
                    self.fb_h.append(tmp_head)
                    self.fb_l.append(tmp_last)
                    self.fb_r.append(tmp_relation)
                self.ok.add((tmp_head, tmp_relation, tmp_last))

def test_model(model, testset, params):
    #saver = tf.train.Saver()
    saver = tf.train.Saver([model.w_ent,model.w_rel,model.w_rel_h,model.w_rel_t,model.b_rel_h,model.b_rel_t,])

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(int(params['start']),int(params['end'])+1):
            st = time.time()
            saver.restore(sess,params['save_fld']+'/model-{}'.format(i*int(params['interval'])))

            lsum = 0
            rsum = 0
            lp_n = 0
            rp_n = 0
            lsum_r = np.zeros(testset.relation_size)
            rsum_r = np.zeros(testset.relation_size)
            lp_n_r = np.zeros(testset.relation_size)
            rp_n_r = np.zeros(testset.relation_size)
            rel_num = np.zeros(testset.relation_size)

            for h,l,rel in zip(testset.fb_h, testset.fb_l, testset.fb_r):
                rel_num[rel] += 1
                nbatches = (testset.entity_size-1) / params['batch_size']
                r_a = rel * np.ones(params['batch_size'])
                t_a = l * np.ones(params['batch_size'])
                a = np.zeros(1)
                for i in xrange(nbatches+1):
                    if i != nbatches:
                        h_a = np.arange(i*params['batch_size'],(i+1)*params['batch_size'])
                    else:
                        h_a = range(i*params['batch_size'],testset.entity_size)
                        h_a.extend(range((i+1)*params['batch_size']-testset.entity_size))
                        h_a = np.array(h_a)
                    tmp_a = sess.run(model.pos,feed_dict = {model.pos_h: h_a, model.pos_t: t_a, model.rel: r_a})
                    a = np.concatenate([a,tmp_a])
                ind = np.argsort(a[1:testset.entity_size+1])
                for i in xrange(testset.entity_size):
                    if ind[i] == h:
                        lsum += i + 1
                        lsum_r[rel] += i + 1
                        if i < 10:
                            lp_n += 1
                            lp_n_r[rel] += 1
                        break
                
                h_a = h * np.ones(params['batch_size'])
                a = np.zeros(1)
                for i in xrange(nbatches+1):
                    if i != nbatches:
                        t_a = np.arange(i*params['batch_size'],(i+1)*params['batch_size'])
                    else:
                        t_a = range(i*params['batch_size'],testset.entity_size)
                        t_a.extend(range((i+1)*params['batch_size']-testset.entity_size))
                        t_a = np.array(t_a)
                    tmp_a = sess.run(model.pos,feed_dict = {model.pos_h: h_a, model.pos_t: t_a, model.rel: r_a})
                    a = np.concatenate([a,tmp_a])
                ind = np.argsort(a[1:testset.entity_size+1])
                for i in xrange(testset.entity_size):
                    if ind[i] == l:
                        rsum += i + 1
                        rsum_r[rel] += i + 1
                        if i < 10:
                            rp_n += 1
                            rp_n_r[rel] += 1
                        break

            print 'left: ', 1.*lsum/len(testset.fb_l), '\t', 1.*lp_n/len(testset.fb_l)
            print 'right: ', 1.*rsum/len(testset.fb_r), '\t', 1.*rp_n/len(testset.fb_r)
            if eval(params['detailed']):
                for i in xrange(testset.relation_size):
                    print testset.relation2id[i]
                    print 'left: ', 1.*lsum_r[i]/rel_num[i], '\t', 1.*lp_n_r[i]/rel_num[i]
                    print 'right: ', 1.*rsum_r[i]/rel_num[i], '\t', 1.*rp_n_r[i]/rel_num[i]
            et = time.time()
            print et-st

        
def main():
    parser = OptionParser()
    parser.add_option("-c", "--conf", dest="configure", help="configure filename")
    options, _ = parser.parse_args() 
    if options.configure:
        conf_file = str(options.configure)
    else:
        print('please specify --conf configure filename')
        exit(-1)
  
    trainset_params, testset_params, net_params, solver_params = process_config(conf_file)
    
    trainset = TrainSet(trainset_params['data_path'], trainset_params['sample'])
    
    net_params['entity_num'] = trainset.entity_num
    net_params['relation_num'] = trainset.relation_num
    net_params['batch_size'] = trainset.record_num / int(net_params['nbatches'])
    model = TransAllModel(net_params)
    model.build_graph()
    
    pretrain = 'pre'
    if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
        pretrain = 'nop'
    if not testset_params.has_key('save_fld') or testset_params['save_fld'] == '':
        testset_params['save_fld'] = 'models/TransAll_v2_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                    trainset_params['data_path'].split('/')[-1], 
                                    trainset_params['sample'],
                                    net_params['embed_size_e'],
                                    net_params['margin'],
                                    net_params['learning_rate'],
                                    net_params['nbatches'],
                                    net_params['normed'],
                                    net_params['activation'],
                                    net_params['opt'],
                                    pretrain)
    print testset_params['save_fld'] 
    testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
    testset = TestSet(trainset_params['data_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = solver_params['gpu_id']
    testset_params['batch_size'] = net_params['batch_size']
    testset_params['activation'] = net_params['activation']
    test_model(model, testset, testset_params)

if __name__ == '__main__':
    main()
