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
from TransX.TransAll_v1 import *
import cPickle as pkl
from sklearn.cluster import KMeans

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def gen_cls_at(entity, head_set, tail_set, nc, var_thresh):
    relation_num = len(head_set)
    norm = np.linalg.norm(entity,axis=1)
    norm = np.maximum(norm,1)
    norm = norm.reshape(entity.shape[0],1)
    X = entity/np.tile(norm, (1,entity.shape[1])) 
    kmeans = KMeans(n_clusters=nc, random_state=0).fit(X)
    cls = kmeans.labels_
    # gen candi_set
    candi_set_head = [set() for _ in xrange(relation_num)]
    candi_set_tail = [set() for _ in xrange(relation_num)]
    for i in xrange(relation_num):
        tmp_cls = set(np.take(cls, list(head_set[i]), axis=0)) 
        tmp_list = []
        for cc in tmp_cls:
            tmp_list.extend(np.where(cls==cc)[0].tolist())
        candi_set_head[i] = set(tmp_list)

        tmp_cls = set(np.take(cls, list(tail_set[i]), axis=0))
        tmp_list = []
        for cc in tmp_cls:
            tmp_list.extend(np.where(cls==cc)[0].tolist())
        candi_set_tail[i] = set(tmp_list)

    dim = X.shape[1]
    rel_head = np.zeros((relation_num,dim))
    rel_tail = np.zeros((relation_num,dim))
    rel_rel = np.zeros((relation_num,dim))
    for i in xrange(relation_num):
        mat_tmp = np.take(entity, list(candi_set_head[i]), axis=0)
        mat_tmp_ = mat_tmp.T - np.mean(mat_tmp, 1)
        var = np.diag(np.dot(mat_tmp_, mat_tmp_.T))
        rel_head[i,:] = var > var_thresh
        
        mat_tmp = np.take(entity, list(candi_set_tail[i]), axis=0)
        mat_tmp_ = mat_tmp.T - np.mean(mat_tmp, 1)
        var = np.diag(np.dot(mat_tmp_, mat_tmp_.T))
        rel_tail[i,:] = var > var_thresh

    rel_rel = (rel_head + rel_tail) > 0
    return candi_set_head, candi_set_tail, rel_head, rel_tail, rel_rel

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
        use_candi = False
        if params['pretrain_model']:
            #w_ent,w_rel = sess.run([model.w_ent,model.w_rel])
            w_ent = sess.run(model.w_ent)
            #print w_ent.shape, w_rel.shape
            #exit(-1)
            saver_p = tf.train.Saver([model.w_ent, model.w_rel])
            #saver_p = tf.train.Saver([model.w_ent])
            saver_p.restore(sess, params['pretrain_model'])
            candi_set_head, candi_set_tail, rel_head, rel_tail, rel_rel = gen_cls_at(w_ent, params['head_set'], params['tail_set'], params['n_clusters'], params['var_thresh'])
            use_candi = True
            #feed_dict = {model.a_rel_h_tmp: np.float32(rel_head), model.a_rel_r_tmp: np.float32(rel_rel), model.a_rel_t_tmp: np.float32(rel_tail)}
            feed_dict = {model.a_rel_h_tmp: np.float32(rel_rel), model.a_rel_r_tmp: np.float32(rel_rel), model.a_rel_t_tmp: np.float32(rel_rel)}
            sess.run(model.reset_at, feed_dict=feed_dict)
            #saver.save(sess, params['save_fld'] + '/model', 0)
            #return
        else:
            relation_num = params['relation_num']
            entity_num = params['entity_num']
            embed_size = int(params['embed_size'])
            feed_dict = {model.a_rel_h_tmp: np.ones((relation_num,embed_size),dtype=np.float32), model.a_rel_r_tmp: np.ones((relation_num,embed_size),dtype=np.float32), model.a_rel_t_tmp: np.ones((relation_num,embed_size),dtype=np.float32)}
            sess.run(model.reset_at, feed_dict=feed_dict)
            #candi_set_head = [set(range(entity_num)) for i in xrange(relation_num)]
            #candi_set_tail = [set(range(entity_num)) for i in xrange(relation_num)]

        total_loss = 0.0
        total_trip_loss = 0.0
        if params['summary_fld']:
            writer = tf.summary.FileWriter(params['summary_fld'], sess.graph)
        initial_step = model.global_step.eval()
        for index in xrange(initial_step, initial_step + int(params['max_iter'])):
            if (index+1) % int(params['at_reset']) == 0:
                w_ent = sess.run(model.w_ent)
                candi_set_head, candi_set_tail, rel_head, rel_tail, rel_rel = gen_cls_at(w_ent, params['head_set'], params['tail_set'], params['n_clusters'], params['var_thresh'])
                use_candi = True
                #feed_dict = {model.a_rel_h_tmp: np.float32(rel_head), model.a_rel_r_tmp: np.float32(rel_rel), model.a_rel_t_tmp: np.float32(rel_tail)}
                feed_dict = {model.a_rel_h_tmp: np.float32(rel_rel), model.a_rel_r_tmp: np.float32(rel_rel), model.a_rel_t_tmp: np.float32(rel_rel)}
                sess.run(model.reset_at, feed_dict=feed_dict)
            
            pos_h, pos_t, neg_h, neg_t, rel = next(batch_gen)
            if use_candi:
                cls_h = np.array([(item[0] not in candi_set_head[item[1]]) for item in zip(neg_h,rel)])
                cls_t = np.array([(item[0] not in candi_set_tail[item[1]]) for item in zip(neg_t,rel)])
                cls = np.zeros_like(rel)
                cls[cls_h] = 1
                cls[cls_t] = 2
                cls = np.float32(cls)
            else:
                cls = np.zeros_like(rel, dtype=np.float32)
            feed_dict = {model.pos_h: pos_h, model.pos_t: pos_t, model.neg_h: neg_h, model.neg_t: neg_t, model.rel: rel, model.cls: cls}
            if params['summary_fld']:
                loss_batch, loss, _, summary = sess.run([model.loss, model.optimizer, model.summary_op], feed_dict=feed_dict)
                writer.add_summary(summary, global_step=index)
            else:
                loss_batch, _ = sess.run([model.loss, model.optimizer], feed_dict=feed_dict)
            total_loss += loss_batch
            if (index + 1) % int(params['interval']) == 0:
                print('Average loss at step {}: {:.8f}'.format(index+1, total_loss / int(params['interval'])))
                total_loss = 0.0
                if params['save_fld']:
                    pkl.dump((candi_set_head, candi_set_tail), open(params['save_fld']+'/ht_candi_{}.pkl'.format(index+1), 'w'))
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
    
    trainset = TrainSet(trainset_params['data_path'], trainset_params['sample'])
    solver_params['head_set'] = trainset.head_set
    solver_params['tail_set'] = trainset.tail_set
    
    net_params['at_reset'] = solver_params['at_reset']
    
    net_params['entity_num'] = trainset.entity_num
    net_params['relation_num'] = trainset.relation_num
    net_params['batch_size'] = trainset.record_num / int(net_params['nbatches'])
    if solver_params['phase'] == 'train':
        model = TransAllModel(net_params)
        model.build_graph()
        os.environ['CUDA_VISIBLE_DEVICES'] = solver_params['gpu_id']
        
        batch_gen = trainset.batch_gen(net_params['batch_size'])
    
        pretrain = 'pre'
        if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
            solver_params['pretrain_model'] = None
            pretrain = 'nop'
        
        if not solver_params.has_key('save_fld'):
            solver_params['save_fld'] = 'models/TransAll_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['activation'],
                                        net_params['alpha'],
                                        net_params['at_reset'],
                                        net_params['opt'],
                                        pretrain)
        elif solver_params['save_fld'] == '':                               
            solver_params['save_fld'] = None
        print solver_params['save_fld']

        if not solver_params.has_key('summary_fld'):
            solver_params['summary_fld']='graphs/TransAll_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['activation'],
                                        net_params['alpha'],
                                        net_params['at_reset'],
                                        net_params['opt'],
                                        pretrain)
        elif solver_params['summary_fld'] == '':
            solver_params['summary_fld'] = None
        
        solver_params['relation_num'] = net_params['relation_num']
        solver_params['entity_num'] = net_params['entity_num']
        solver_params['embed_size'] = net_params['embed_size_e']
        solver_params['n_clusters'] = int(net_params['n_clusters'])
        solver_params['var_thresh'] = float(net_params['var_thresh'])
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
        models = []
        for i in xrange(4):
            with tf.device('/gpu:%d'%i):
                models.append(TransAllModel(net_params))
                models[i].build_graph()
                tf.get_variable_scope().reuse_variables()
        pretrain = 'pre'
        if not solver_params.has_key('pretrain_model') or solver_params['pretrain_model'] == '':
            pretrain = 'nop'
        if not testset_params.has_key('save_fld') or testset_params['save_fld'] == '':
            testset_params['save_fld'] = 'models/TransAll_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                                        trainset_params['data_path'].split('/')[-1], 
                                        trainset_params['sample'],
                                        net_params['embed_size_e'],
                                        net_params['margin'],
                                        net_params['learning_rate'],
                                        net_params['nbatches'],
                                        net_params['normed'],
                                        net_params['activation'],
                                        net_params['alpha'],
                                        net_params['at_reset'],
                                        net_params['opt'],
                                        pretrain)
        print testset_params['save_fld'] 
        testset_params['dataset'] = trainset_params['data_path'].split('/')[-1]
        testset = TestSet(trainset_params['data_path'], 'test')
        testset_params['batch_size'] = net_params['batch_size']
        testset_params['activation'] = net_params['activation']
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
