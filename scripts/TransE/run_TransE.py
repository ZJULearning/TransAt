import numpy as np
import tensorflow as tf
import os
import time
from optparse import OptionParser

import sys
sys.path.append('./')
from utils.trainset import TrainSet
from utils.testset import TestSet
from utils.testset import test_model_link
from utils.process_config import *
from TransX.TransE import *

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def train_model(model, batch_gen, params):
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
            saver_p = tf.train.Saver([model.w_ent, model.w_rel])
            saver_p.restore(sess, params['pretrain_model'])
            w_ent, w_rel = sess.run([model.w_ent, model.w_rel])
            #s, u, v = sess.run(tf.svd(model._normed(model.w_ent), full_matrices=True))
            #w_rel = np.dot(w_rel, v)
            u, s, v = np.linalg.svd(w_ent)
            w_rel = np.dot(w_rel, v.T)
            #print s, s1
            s_ = np.zeros((u.shape[0],v.shape[0]))
            s = np.diag(s)
            s_[:s.shape[0],:s.shape[0]] = s
            #print np.sum(np.abs(w_ent - np.dot(np.dot(u,s_),v.T)))
            #exit(-1)
            w_ent = np.dot(u, s_)
            init_op = model.w_ent.assign(w_ent)
            sess.run(init_op)
            init_op = model.w_rel.assign(w_rel)
            sess.run(init_op)
            #saver.save(sess, params['save_fld'] + '/model', 0)
            #return 0
            #print 'bugbug'
            if eval(params['dorc']):
                w_ent_t = sess.run(model.w_ent)
                init_op = model.w_ent_t.assign(w_ent_t)
                sess.run(init_op)

        total_loss = 0.0
        if params['summary_fld']:
            writer = tf.summary.FileWriter(params['summary_fld'], sess.graph)
        initial_step = model.global_step.eval()
        ts = time.time()
        for index in xrange(initial_step, initial_step + int(params['max_iter'])):
            pos_h, pos_t, neg_h, neg_t, rel = next(batch_gen)
            feed_dict = {model.pos_h: pos_h, model.pos_t: pos_t, model.neg_h: neg_h, model.neg_t: neg_t, model.rel: rel}
            if params['summary_fld']:
                loss_batch, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
                writer.add_summary(summary, global_step=index)
            else:
                loss_batch, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
            total_loss += loss_batch
            if (index + 1) % int(params['interval']) == 0:
                te = time.time()
                print('Average loss at step {}: {:.8f}, at: {}'.format(index+1, total_loss / int(params['interval']), (te-ts)/int(params['interval'])))
                total_loss = 0.0
                if params['save_fld']:
                    saver.save(sess, params['save_fld'] + '/model', index+1)
                ts = time.time()
        
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
    
    #trainset = TrainSet(trainset_params['data_path'], trainset_params['sample'], asym=True)
    trainset = TrainSet(trainset_params['data_path'], trainset_params['sample'])
    
    net_params['entity_num'] = trainset.entity_num
    net_params['relation_num'] = trainset.relation_num
    net_params['batch_size'] = trainset.record_num / int(net_params['nbatches'])
    
    if solver_params['phase'] == 'train':
        model = TransEModel(net_params)
        model.build_graph()
        os.environ['CUDA_VISIBLE_DEVICES'] = solver_params['gpu_id']
        batch_gen = trainset.batch_gen(net_params['batch_size'])
    
        if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
            solver_params['pretrain_model'] = None
        
        if not solver_params.has_key('save_fld'):
            solver_params['save_fld'] = 'models/TransE_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['dorc'],
                                        net_params['opt'])
        elif solver_params['save_fld'] == '':                               
            solver_params['save_fld'] = None
        print solver_params['save_fld']

        if not solver_params.has_key('summary_fld'):
            solver_params['summary_fld']='graphs/TransE_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['dorc'],
                                        net_params['opt'])
        elif solver_params['summary_fld'] == '':
            solver_params['summary_fld'] = None

        solver_params['dorc'] = net_params['dorc']
        train_model(model, batch_gen, solver_params)
        
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
        #models = TransEModel(net_params)
        #models.build_graph()
        #os.environ['CUDA_VISIBLE_DEVICES'] = '3'
        models = []
        for i in xrange(4):
            with tf.device('/gpu:%d'%i):
                models.append(TransEModel(net_params))
                models[i].build_graph()
                tf.get_variable_scope().reuse_variables()
        if not testset_params.has_key('save_fld') or testset_params['save_fld'] == '':
            testset_params['save_fld'] = 'models/TransE_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['dorc'],
                                        net_params['opt'])
        
        print testset_params['save_fld']
        testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
        testset = TestSet(trainset_params['data_path'], 'test')
        #testset = TestSet(trainset_params['data_path'], 'train')
        testset_params['batch_size'] = net_params['batch_size']
        if testset_params['testtype'] == 'link':
            test_model_link(models, testset, testset_params)
        elif testset_params['testtype'] == 'trip':
            raise ValueError('Wait to finish.')
        else:
            raise ValueError('Undefined testtype.')
    else:
        raise ValueError('Undefined phase.')

if __name__ == '__main__':
    main()
