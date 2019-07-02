import numpy as np
import tensorflow as tf
import os
import time
from optparse import OptionParser

import sys
sys.path.append('./')
from utils.trainset import TrainSet
from utils.testset import *
from utils.process_config import *
from TransX.TransC import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def cal_center(entity, c_set):
    num = len(c_set)
    dim = entity.shape[1]
    center = np.zeros((num,dim))
    for i in xrange(num):
        center[i] = np.mean(np.take(entity,c_set[i],axis=0),0)
    return center

def train_model(model, trainset, params):
    batch_gen = trainset.batch_gen(params['batch_size'])
    saver = tf.train.Saver(max_to_keep=0)
    #saver = tf.train.Saver(max_to_keep=1)

    initial_step = 0
    try:
        os.mkdir(params['save_fld'])
    except Exception as e:
        pass
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        if params['pretrain_model']:
            #w_ent,w_rel = sess.run([model.w_ent,model.w_rel])
            #print w_ent.shape, w_rel.shape
            #exit(-1)
            #saver_p = tf.train.Saver([model.w_ent, model.w_rel])
            saver_p = tf.train.Saver([model.w_ent,model.w_rel])
            saver_p.restore(sess, params['pretrain_model'])
            w_ent,w_rel = sess.run([model.w_ent,model.w_rel])
            w_ent = w_ent / np.linalg.norm(w_ent,axis=1).reshape((-1,1))
            w_rel = w_rel / np.linalg.norm(w_rel,axis=1).reshape((-1,1))
            rel_c_h = cal_center(w_ent, trainset.r_e_h)
            rel_c_t = cal_center(w_ent, trainset.r_e_t)
            #rel_c_h = np.zeros((18,50))
            #rel_c_t = np.zeros((18,50))
            feed_dict_reset_center = {model.rel_c_h_tmp:rel_c_h, model.rel_c_t_tmp:rel_c_t}
            sess.run(model.reset_center, feed_dict=feed_dict_reset_center)
            w_rel = w_rel + rel_c_h - rel_c_t
            #print np.linalg.norm(rel_c_h,axis=1)
            #print np.linalg.norm(rel_c_t,axis=1)
            #print np.linalg.norm(w_rel,axis=1)
            #exit(-1)
            #s_rel = np.identity(w_rel.shape[1]) 
            s_rel = np.tile(np.identity(w_rel.shape[1])[np.newaxis,:],(w_rel.shape[0],1,1))
            init_r = model.w_rel.assign(w_rel)
            init_s_h = model.s_rel_h.assign(s_rel)
            init_s_t = model.s_rel_t.assign(s_rel)
            sess.run([init_r,init_s_h,init_s_t])
            #saver.save(sess, params['save_fld'] + '/model', 0)
            #return 0

        total_loss = 0.0
        if params['summary_fld']:
            writer = tf.summary.FileWriter(params['summary_fld'], sess.graph)
        initial_step = model.global_step.eval()
        for index in xrange(initial_step, initial_step + int(params['max_iter'])):
            if index % int(params['center_reset']) == 0:
            #if True:
                #print index
                entity = sess.run(model.w_ent)
                entity = entity / np.linalg.norm(entity,axis=1).reshape((-1,1))
                rel_c_h = cal_center(entity, trainset.r_e_h)
                rel_c_t = cal_center(entity, trainset.r_e_t)
                feed_dict_reset_center = {model.rel_c_h_tmp:rel_c_h, model.rel_c_t_tmp:rel_c_t}
                sess.run(model.reset_center, feed_dict=feed_dict_reset_center)
            pos_h, pos_t, neg_h, neg_t, rel = next(batch_gen)
            feed_dict = {model.pos_h: pos_h, model.pos_t: pos_t, model.neg_h: neg_h, model.neg_t: neg_t, model.rel: rel}
            if params['summary_fld']:
                loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
                writer.add_summary(summary, global_step=index)
            else:
                loss_batch, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
            total_loss += loss_batch
            if (index + 1) % int(params['interval']) == 0:
                print('Average loss at step {}: {:.8f}'.format(index+1, total_loss / int(params['interval'])))
                total_loss = 0.0
                if params['save_fld']:
                    saver.save(sess, params['save_fld'] + '/model', index+1)
        
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
    
    trainset = TrainSet(trainset_params['data_path'], trainset_params['sample'], True)
    
    net_params['entity_num'] = trainset.entity_num
    net_params['relation_num'] = trainset.relation_num
    net_params['batch_size'] = trainset.record_num / int(net_params['nbatches'])
    model = TransCModel(net_params)
    model.build_graph()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = solver_params['gpu_id']
    if solver_params['phase'] == 'train':
        pretrain = 'pre'
        if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
            solver_params['pretrain_model'] = None
            pretrain = 'nop'
        
        if not solver_params.has_key('save_fld'):
            solver_params['save_fld'] = 'models/TransC_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['embed_size_r'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['opt'],
                                        pretrain)
        elif solver_params['save_fld'] == '':                               
            solver_params['save_fld'] = None
        print solver_params['save_fld']

        if not solver_params.has_key('summary_fld'):
            solver_params['summary_fld']='graphs/TransC_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['embed_size_r'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['opt'],
                                        pretrain)
        elif solver_params['summary_fld'] == '':
            solver_params['summary_fld'] = None

        solver_params['batch_size'] = net_params['batch_size']
        train_model(model, trainset, solver_params)
        
        if solver_params['save_fld']:
            testset_params['save_fld'] = solver_params['save_fld']
            testset_params['start'] = 1
            testset_params['end'] = 1
            testset_params['interval'] = solver_params['max_iter']
            testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
            testset = TestSet(trainset_params['data_path'], 'test')
            testset_params['batch_size'] = net_params['batch_size']
            if testset_params['testtype'] == 'link':
                test_model_link(model, testset, testset_params)
            elif testset_params['testtype'] == 'trip':
                raise ValueError('Wait to finish.')
            else:
                raise ValueError('Undefined testtype.')
    elif solver_params['phase'] == 'val':
        raise ValueError('Wait to finish.')
    elif solver_params['phase'] == 'test':
        pretrain = 'pre'
        if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
            pretrain = 'nop'
        if not testset_params.has_key('save_fld') or testset_params['save_fld'] == '':
            testset_params['save_fld'] = 'models/TransC_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['embed_size_r'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['opt'],
                                        pretrain)
        print testset_params['save_fld'] 
        testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
        testset = TestSet(trainset_params['data_path'], 'test')
        testset_params['batch_size'] = net_params['batch_size']
        if testset_params['testtype'] == 'link':
            test_model_link(model, testset, testset_params)
        elif testset_params['testtype'] == 'trip':
            raise ValueError('Wait to finish.')
        else:
            raise ValueError('Undefined testtype.')
    else:
        raise ValueError('Undefined phase.')

if __name__ == '__main__':
    main()
