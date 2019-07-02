import numpy as np
import tensorflow as tf
import os
import time
from optparse import OptionParser

import sys
sys.path.append('./')
from utils.trainset import TrainSet
from utils.process_config import *
from TransX.TransR import *

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

    def test(self, weights_relation, weights_entity, weights_A):
        n = weights_entity.shape[1]
        lsum = 0
        rsum = 0
        lp_n = 0
        rp_n = 0
        
        weights_A_entity = np.zeros((self.relation_size,self.entity_size,n))
        for i in xrange(self.relation_size):
            weights_A_entity[i] = np.dot(weights_A[i], weights_entity.T).T

        for h,l,rel in zip(self.fb_h, self.fb_l, self.fb_r):
            prd_h = (weights_A_entity[rel,l] - weights_relation[rel]).reshape((1,n))
            a = np.sum(np.abs(prd_h - weights_A_entity[rel]),1)
            ind = np.argsort(a)
            for i in xrange(self.entity_size):
                if ind[i] == h:
                    lsum += i + 1
                    if i < 10: lp_n += 1
                    break
            
            prd_l = (weights_A_entity[rel,h] + weights_relation[rel]).reshape((1,n))
            a = np.sum(np.abs(weights_A_entity[rel] - prd_l),1)
            ind = np.argsort(a)
            for i in xrange(self.entity_size):
                if ind[i] == l:
                    rsum += i + 1
                    if i < 10: rp_n += 1
                    break

        print 'left: ', 1.*lsum/len(self.fb_l), '\t', 1.*lp_n/len(self.fb_l)
        print 'right: ', 1.*rsum/len(self.fb_r), '\t', 1.*rp_n/len(self.fb_r)

def test_model(model, testset, params):
    saver = tf.train.Saver()

    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for i in xrange(int(params['start']),int(params['end'])+1):
            saver.restore(sess,params['save_fld']+'/model-{}'.format(i*int(params['interval'])))

            entity, relation, mat = sess.run([model.w_ent, model.w_rel, model.m_rel])
            if eval(params['normed']):
                norm = np.linalg.norm(entity,axis=1)
                norm = np.maximum(norm,1)
                norm = norm.reshape(entity.shape[0],1)
                entity = entity/np.tile(norm, (1,entity.shape[1]))
                norm = np.linalg.norm(relation,axis=1)
                norm = np.maximum(norm,1)
                norm = norm.reshape(relation.shape[0],1)
                relation = relation/np.tile(norm, (1,relation.shape[1]))

            testset.test(relation, entity, mat)    
        
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
    model = TransRModel(net_params)
    model.build_graph()
    
    pretrain = 'pre'
    if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
        pretrain = 'nop'
    if not testset_params.has_key('save_fld') or testset_params['save_fld'] == '':
        testset_params['save_fld'] = 'models/TransR_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                    trainset_params['data_path'].split('/')[-1], 
                                    trainset_params['sample'],
                                    net_params['embed_size_e'],
                                    net_params['margin'],
                                    net_params['learning_rate'],
                                    net_params['nbatches'],
                                    net_params['normed'],
                                    net_params['opt'],
                                    pretrain)
    
    testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
    testset = TestSet(trainset_params['data_path'])
    os.environ['CUDA_VISIBLE_DEVICES'] = solver_params['gpu_id']
    test_model(model, testset, testset_params)

if __name__ == '__main__':
    main()
